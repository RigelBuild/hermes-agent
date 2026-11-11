package main

import (
	"bytes"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
)

const (
	// maxReleaseAssets is GitHub's per-release asset limit.
	maxReleaseAssets = 1000

	nixCacheInfo   = "nix-cache-info"
	releaseKeyName = "nix-release-cache.pub"
	platformName   = "platform"
	pathInfoSuffix = ".path-info.json"
)

// storeHashPattern matches a store path hash: 32 characters of nix base32.
var storeHashPattern = regexp.MustCompile(`^[0-9a-df-np-sv-z]{32}$`)

// upstreamChecker reports whether another binary cache already serves a path.
type upstreamChecker interface {
	has(hash string) (bool, error)
}

type assembleConfig struct {
	platformDirs []string
	outDir       string
	upstream     upstreamChecker // nil keeps every path
	maxAssets    int
}

// asset is one release file. Exactly one of data or src is set.
type asset struct {
	data []byte
	src  string
}

func (a asset) digest() ([]byte, error) {
	if a.src == "" {
		sum := sha256.Sum256(a.data)
		return sum[:], nil
	}
	f, err := os.Open(a.src)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return nil, err
	}
	return h.Sum(nil), nil
}

// plan maps release asset names to contents. Adding the same name twice is
// fine when the contents match, and an error when they do not.
type plan struct {
	assets map[string]asset
}

func (p plan) add(name string, a asset) error {
	prev, ok := p.assets[name]
	if !ok {
		p.assets[name] = a
		return nil
	}
	want, err := prev.digest()
	if err != nil {
		return err
	}
	got, err := a.digest()
	if err != nil {
		return err
	}
	if !bytes.Equal(want, got) {
		return fmt.Errorf("conflicting contents for release asset %s", name)
	}
	return nil
}

// platformDir is one downloaded build artifact: the output of nix copy under
// cache/, plus the metadata the release needs.
type platformDir struct {
	name      string
	pathInfo  string
	pub       []byte
	cacheInfo []byte
}

func readPlatform(dir string) (platformDir, error) {
	raw, err := os.ReadFile(filepath.Join(dir, platformName))
	if err != nil {
		return platformDir{}, err
	}
	name := strings.TrimSpace(string(raw))
	if name == "" {
		return platformDir{}, fmt.Errorf("%s: empty platform name", dir)
	}
	pathInfos, err := filepath.Glob(filepath.Join(dir, "*"+pathInfoSuffix))
	if err != nil {
		return platformDir{}, err
	}
	if len(pathInfos) != 1 {
		return platformDir{}, fmt.Errorf("%s: want one %s file, found %d", dir, pathInfoSuffix, len(pathInfos))
	}
	if filepath.Base(pathInfos[0]) != name+pathInfoSuffix {
		return platformDir{}, fmt.Errorf("%s: %s does not match platform %s", dir, filepath.Base(pathInfos[0]), name)
	}
	pub, err := os.ReadFile(filepath.Join(dir, releaseKeyName))
	if err != nil {
		return platformDir{}, err
	}
	cacheInfo, err := os.ReadFile(filepath.Join(dir, "cache", nixCacheInfo))
	if err != nil {
		return platformDir{}, err
	}
	return platformDir{name: name, pathInfo: pathInfos[0], pub: pub, cacheInfo: cacheInfo}, nil
}

// storeHash returns the hash part of a store path, which names its narinfo.
func storeHash(storePath string) (string, error) {
	base := filepath.Base(storePath)
	hash, _, ok := strings.Cut(base, "-")
	if !ok || !storeHashPattern.MatchString(hash) {
		return "", fmt.Errorf("store path %q has no 32-character base32 hash", storePath)
	}
	return hash, nil
}

// assemble merges platform directories into one flat release asset set in
// cfg.outDir. A store path already seen on an earlier platform is kept once,
// and it is an error only when its identity (NarHash, NarSize, References)
// differs. It fails before writing when the asset count exceeds cfg.maxAssets.
// It returns the number of assets written.
func assemble(cfg assembleConfig) (int, error) {
	if len(cfg.platformDirs) == 0 {
		return 0, errors.New("no platform directories given")
	}
	if entries, err := os.ReadDir(cfg.outDir); err == nil && len(entries) > 0 {
		return 0, fmt.Errorf("%s is not empty", cfg.outDir)
	}

	p := plan{assets: map[string]asset{}}
	identities := map[string]narIdentity{}
	seen := map[string]bool{}
	for _, dir := range cfg.platformDirs {
		pl, err := readPlatform(dir)
		if err != nil {
			return 0, err
		}
		if seen[pl.name] {
			return 0, fmt.Errorf("platform %s given twice", pl.name)
		}
		seen[pl.name] = true
		if err := p.add(nixCacheInfo, asset{data: pl.cacheInfo}); err != nil {
			return 0, fmt.Errorf("%s: %w", dir, err)
		}
		if err := p.add(releaseKeyName, asset{data: pl.pub}); err != nil {
			return 0, fmt.Errorf("%s: %w", dir, err)
		}
		if err := p.add(pl.name+pathInfoSuffix, asset{src: pl.pathInfo}); err != nil {
			return 0, fmt.Errorf("%s: %w", dir, err)
		}

		nars, err := filepath.Glob(filepath.Join(dir, "cache", "nar", "*"))
		if err != nil {
			return 0, err
		}
		if len(nars) == 0 {
			return 0, fmt.Errorf("platform %s has no NAR files", pl.name)
		}
		narinfos, err := filepath.Glob(filepath.Join(dir, "cache", "*.narinfo"))
		if err != nil {
			return 0, err
		}
		if len(narinfos) == 0 {
			return 0, fmt.Errorf("platform %s has no narinfo files", pl.name)
		}

		for _, path := range narinfos {
			data, err := os.ReadFile(path)
			if err != nil {
				return 0, err
			}
			ref, err := parseNarinfo(data)
			if err != nil {
				return 0, fmt.Errorf("%s: %w", path, err)
			}
			hash, err := storeHash(ref.storePath)
			if err != nil {
				return 0, fmt.Errorf("%s: %w", path, err)
			}
			if prev, ok := identities[ref.storePath]; ok {
				if prev != ref.identity {
					return 0, fmt.Errorf("%s: %s has a different NarHash, NarSize or References on another platform", path, ref.storePath)
				}
				continue
			}
			identities[ref.storePath] = ref.identity
			if cfg.upstream != nil {
				cached, err := cfg.upstream.has(hash)
				if err != nil {
					return 0, fmt.Errorf("%s: %w", path, err)
				}
				if cached {
					continue
				}
			}

			flat, nar, err := flattenNarinfo(data)
			if err != nil {
				return 0, fmt.Errorf("%s: %w", path, err)
			}
			narSrc := filepath.Join(dir, "cache", "nar", nar)
			if _, err := os.Stat(narSrc); err != nil {
				return 0, fmt.Errorf("%s: %w", path, err)
			}
			if err := p.add(hash+".narinfo", asset{data: flat}); err != nil {
				return 0, fmt.Errorf("%s: %w", path, err)
			}
			if err := p.add(nar, asset{src: narSrc}); err != nil {
				return 0, fmt.Errorf("%s: %w", path, err)
			}
		}
	}

	if len(p.assets) > cfg.maxAssets {
		return 0, fmt.Errorf("release has %d assets; GitHub allows %d per release", len(p.assets), cfg.maxAssets)
	}

	names := make([]string, 0, len(p.assets))
	for name := range p.assets {
		names = append(names, name)
	}
	slices.Sort(names)
	if err := os.MkdirAll(cfg.outDir, 0o755); err != nil {
		return 0, err
	}
	for _, name := range names {
		a := p.assets[name]
		dst := filepath.Join(cfg.outDir, name)
		if a.src == "" {
			if err := os.WriteFile(dst, a.data, 0o644); err != nil {
				return 0, err
			}
			continue
		}
		if err := copyFile(a.src, dst); err != nil {
			return 0, err
		}
	}
	return len(names), nil
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	return out.Close()
}
