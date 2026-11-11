package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const (
	sharedHash = "1b9p3x1z5r7c2m8k4w6v0q9s3d5f7h1j"
	linuxHash  = "6v4d2c8b0ssfmzqk3p9lhrx5j7wn1dyg"
	darwinHash = "xk1zq8r2m4c6b0d3f5g7h9jlnpsvwxy0"

	testPub = "nix-release-cache-1:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=\n"
)

// fixturePath is one store path in a fixture platform: its narinfo and NAR.
type fixturePath struct {
	hash, nar, narHash, narSize string
}

// fixturePlatform is one build artifact. paths[0] is the built output.
type fixturePlatform struct {
	name  string
	paths []fixturePath
}

var (
	shared     = fixturePath{hash: sharedHash, nar: "shared.nar.xz", narHash: "sha256:shared", narSize: "100"}
	linuxOnly  = fixturePath{hash: linuxHash, nar: "linux.nar.xz", narHash: "sha256:linux", narSize: "200"}
	darwinOnly = fixturePath{hash: darwinHash, nar: "darwin.nar.xz", narHash: "sha256:darwin", narSize: "300"}

	linuxPlatform  = fixturePlatform{name: "hermes-agent-v2026.11.1-rigel.1-linux-x64", paths: []fixturePath{linuxOnly, shared}}
	darwinPlatform = fixturePlatform{name: "hermes-agent-v2026.11.1-rigel.1-darwin-arm64", paths: []fixturePath{darwinOnly, shared}}
)

func narinfoFor(p fixturePath) string {
	return "StorePath: /nix/store/" + p.hash + "-hermes-agent\n" +
		"URL: nar/" + p.nar + "\n" +
		"Compression: xz\n" +
		"FileHash: sha256:file-" + p.nar + "\n" +
		"FileSize: 42\n" +
		"NarHash: " + p.narHash + "\n" +
		"NarSize: " + p.narSize + "\n" +
		"References: \n" +
		"Deriver: " + p.hash + "-hermes-agent.drv\n" +
		"Sig: cache-release-1:AAAA\n"
}

// buildPlatform writes one platform directory in the layout that
// `nix copy` plus the release workflow produce, and returns its path.
func buildPlatform(t *testing.T, root string, fp fixturePlatform, pub string) string {
	t.Helper()
	dir := filepath.Join(root, fp.name)
	mustWrite(t, filepath.Join(dir, "platform"), fp.name+"\n")
	mustWrite(t, filepath.Join(dir, releaseKeyName), pub)
	mustWrite(t, filepath.Join(dir, fp.name+pathInfoSuffix), `{"/nix/store/`+fp.paths[0].hash+`-hermes-agent": {}}`)
	mustWrite(t, filepath.Join(dir, "cache", nixCacheInfo), "StoreDir: /nix/store\nWantMassQuery: 1\nPriority: 30\n")
	for _, p := range fp.paths {
		mustWrite(t, filepath.Join(dir, "cache", p.hash+".narinfo"), narinfoFor(p))
		mustWrite(t, filepath.Join(dir, "cache", "nar", p.nar), "nar:"+p.nar)
	}
	return dir
}

func mustWrite(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func wantErr(t *testing.T, err error, substr string) {
	t.Helper()
	if err == nil {
		t.Fatalf("want error containing %q, got nil", substr)
	}
	if !strings.Contains(err.Error(), substr) {
		t.Fatalf("error = %q, want it to contain %q", err, substr)
	}
}

// fakeUpstream reports a store hash as cached when it is a key set to true.
type fakeUpstream map[string]bool

func (f fakeUpstream) has(hash string) (bool, error) { return f[hash], nil }

func TestAssembleFlattensAndDedupes(t *testing.T) {
	root := t.TempDir()
	linux := buildPlatform(t, root, linuxPlatform, testPub)
	darwin := buildPlatform(t, root, darwinPlatform, testPub)
	out := filepath.Join(root, "release")

	n, err := assemble(assembleConfig{platformDirs: []string{linux, darwin}, outDir: out, maxAssets: maxReleaseAssets})
	if err != nil {
		t.Fatal(err)
	}

	want := []string{
		nixCacheInfo, releaseKeyName,
		linuxPlatform.name + pathInfoSuffix, darwinPlatform.name + pathInfoSuffix,
		linuxHash + ".narinfo", darwinHash + ".narinfo", sharedHash + ".narinfo",
		"linux.nar.xz", "darwin.nar.xz", "shared.nar.xz",
	}
	if n != len(want) {
		t.Fatalf("wrote %d assets, want %d", n, len(want))
	}
	entries, err := os.ReadDir(out)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != len(want) {
		t.Fatalf("release dir has %d entries, want %d", len(entries), len(want))
	}
	for _, name := range want {
		info, err := os.Stat(filepath.Join(out, name))
		if err != nil {
			t.Errorf("missing %s: %v", name, err)
			continue
		}
		if info.IsDir() {
			t.Errorf("%s is a directory; release assets must be flat", name)
		}
	}

	narinfo, err := os.ReadFile(filepath.Join(out, sharedHash+".narinfo"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(narinfo), "\nURL: shared.nar.xz\n") {
		t.Errorf("shared narinfo not flattened:\n%s", narinfo)
	}
	if strings.Contains(string(narinfo), "nar/") {
		t.Errorf("shared narinfo still points under nar/:\n%s", narinfo)
	}
}

func TestAssembleRejectsConflicts(t *testing.T) {
	changedShared := shared
	changedShared.narHash = "sha256:changed"
	changedPub := "nix-release-cache-1:BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB=\n"

	tests := map[string]struct {
		darwinPlatform fixturePlatform
		darwinPub      string
		wantErr        string
	}{
		"same path, different NarHash": {
			darwinPlatform: fixturePlatform{name: darwinPlatform.name, paths: []fixturePath{darwinOnly, changedShared}},
			darwinPub:      testPub,
			wantErr:        "different NarHash",
		},
		"different release key": {
			darwinPlatform: darwinPlatform,
			darwinPub:      changedPub,
			wantErr:        "conflicting contents for release asset nix-release-cache.pub",
		},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			root := t.TempDir()
			linux := buildPlatform(t, root, linuxPlatform, testPub)
			darwin := buildPlatform(t, root, tc.darwinPlatform, tc.darwinPub)
			_, err := assemble(assembleConfig{platformDirs: []string{linux, darwin}, outDir: filepath.Join(root, "out"), maxAssets: maxReleaseAssets})
			wantErr(t, err, tc.wantErr)
		})
	}
}

func TestAssembleRejectsPlatformWithoutNAR(t *testing.T) {
	root := t.TempDir()
	dir := buildPlatform(t, root, linuxPlatform, testPub)
	if err := os.RemoveAll(filepath.Join(dir, "cache", "nar")); err != nil {
		t.Fatal(err)
	}
	_, err := assemble(assembleConfig{platformDirs: []string{dir}, outDir: filepath.Join(root, "out"), maxAssets: maxReleaseAssets})
	wantErr(t, err, "has no NAR files")
}

func TestAssembleFailsBeforeWritingOverAssetLimit(t *testing.T) {
	root := t.TempDir()
	linux := buildPlatform(t, root, linuxPlatform, testPub)
	out := filepath.Join(root, "out")

	_, err := assemble(assembleConfig{platformDirs: []string{linux}, outDir: out, maxAssets: 3})
	wantErr(t, err, "GitHub allows 3 per release")
	if _, statErr := os.Stat(out); !os.IsNotExist(statErr) {
		t.Fatalf("output dir written despite asset limit: %v", statErr)
	}
}

func TestAssembleSkipsPathsCachedUpstream(t *testing.T) {
	root := t.TempDir()
	linux := buildPlatform(t, root, linuxPlatform, testPub)
	darwin := buildPlatform(t, root, darwinPlatform, testPub)
	out := filepath.Join(root, "release")

	n, err := assemble(assembleConfig{
		platformDirs: []string{linux, darwin},
		outDir:       out,
		upstream:     fakeUpstream{sharedHash: true},
		maxAssets:    maxReleaseAssets,
	})
	if err != nil {
		t.Fatal(err)
	}
	// The shared path and its NAR are served upstream, so they are dropped.
	// Both platforms still ship their own output, key, cache-info and metadata.
	if n != 8 {
		t.Fatalf("wrote %d assets, want 8", n)
	}
	for _, name := range []string{sharedHash + ".narinfo", "shared.nar.xz"} {
		if _, err := os.Stat(filepath.Join(out, name)); !os.IsNotExist(err) {
			t.Errorf("%s should be skipped, stat err = %v", name, err)
		}
	}
}
