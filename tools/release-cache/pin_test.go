package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

const pinStorePath = "/nix/store/1b9p3x1z5r7c2m8k4w6v0q9s3d5f7h1j-hermes-agent"

func TestWritePinRecordsTagAndStorePath(t *testing.T) {
	dir := t.TempDir()
	info := filepath.Join(dir, "path-info.json")
	mustWrite(t, info, `{"`+pinStorePath+`": {"narSize": 100}}`)
	out := filepath.Join(dir, "nix", "release-closure", "x86_64-linux.json")

	if err := writePin("x86_64-linux", "v2026.11.1-rigel.1", info, out); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	var got pin
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatal(err)
	}
	want := pin{Tag: "v2026.11.1-rigel.1", StorePath: pinStorePath}
	if got != want {
		t.Fatalf("pin = %+v, want %+v", got, want)
	}
}

func TestWritePinRejectsBadInput(t *testing.T) {
	one := `{"` + pinStorePath + `": {}}`
	two := `{"` + pinStorePath + `": {}, "/nix/store/6v4d2c8b0ssfmzqk3p9lhrx5j7wn1dyg-other": {}}`
	tests := map[string]struct {
		system, tag, info, wantErr string
	}{
		"unknown system":   {system: "riscv64-linux", tag: "v1.0-rigel.1", info: one, wantErr: "unknown system"},
		"tag without v":    {system: "x86_64-linux", tag: "1.0-rigel.1", info: one, wantErr: "does not match"},
		"tag not rigel":    {system: "x86_64-linux", tag: "v1.0", info: one, wantErr: "does not match"},
		"two store paths":  {system: "x86_64-linux", tag: "v1.0-rigel.1", info: two, wantErr: "exactly one store path"},
		"not a store path": {system: "x86_64-linux", tag: "v1.0-rigel.1", info: `{"/tmp/x": {}}`, wantErr: "is not a store path"},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			info := filepath.Join(dir, "path-info.json")
			mustWrite(t, info, tc.info)
			err := writePin(tc.system, tc.tag, info, filepath.Join(dir, "pin.json"))
			wantErr(t, err, tc.wantErr)
		})
	}
}
