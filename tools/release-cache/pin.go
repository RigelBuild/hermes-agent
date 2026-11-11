package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
)

// releaseSystems are the systems the release workflow builds.
var releaseSystems = []string{"x86_64-linux", "aarch64-linux", "aarch64-darwin"}

// tagPattern matches the tags the release workflow runs on.
var tagPattern = regexp.MustCompile(`^v[A-Za-z0-9._+-]*-rigel\.[A-Za-z0-9._+-]+$`)

const storeDir = "/nix/store/"

// pin is nix/release-closure/<system>.json. The Nix side reads it with
// builtins.fromJSON and fetches storePath from the release tag.
type pin struct {
	Tag       string `json:"tag"`
	StorePath string `json:"storePath"`
}

// writePin reads the single store path from a `nix path-info --json` file and
// writes the pin for system. It replaces any existing pin.
func writePin(system, tag, pathInfoFile, outFile string) error {
	if !slices.Contains(releaseSystems, system) {
		return fmt.Errorf("unknown system %q; want one of %s", system, strings.Join(releaseSystems, ", "))
	}
	if !tagPattern.MatchString(tag) {
		return fmt.Errorf("tag %q does not match %s", tag, tagPattern)
	}
	data, err := os.ReadFile(pathInfoFile)
	if err != nil {
		return err
	}
	var infos map[string]json.RawMessage
	if err := json.Unmarshal(data, &infos); err != nil {
		return fmt.Errorf("%s: want nix path-info --json object: %w", pathInfoFile, err)
	}
	if len(infos) != 1 {
		return fmt.Errorf("%s: want exactly one store path, found %d", pathInfoFile, len(infos))
	}
	var storePath string
	for sp := range infos {
		storePath = sp
	}
	if rest, ok := strings.CutPrefix(storePath, storeDir); !ok || rest == "" || strings.Contains(rest, "/") {
		return fmt.Errorf("%s: %q is not a store path", pathInfoFile, storePath)
	}

	out, err := json.MarshalIndent(pin{Tag: tag, StorePath: storePath}, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(outFile), 0o755); err != nil {
		return err
	}
	return os.WriteFile(outFile, append(out, '\n'), 0o644)
}
