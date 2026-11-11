package main

import (
	"strings"
	"testing"
)

const sampleNarinfo = `StorePath: /nix/store/1b9p3x1z5r7c2m8k4w6v0q9s3d5f7h1j-hermes-agent
URL: nar/abc123.nar.xz
Compression: xz
FileHash: sha256:abc
FileSize: 42
NarHash: sha256:def
NarSize: 100
References: 6v4d2c8b0ssfmzqk3p9lhrx5j7wn1dyg-dep-b 1b9p3x1z5r7c2m8k4w6v0q9s3d5f7h1j-dep-a
Deriver: 1b9p3x1z5r7c2m8k4w6v0q9s3d5f7h1j-hermes-agent.drv
Sig: cache-release-1:AAAA
`

func TestParseNarinfoReadsIdentity(t *testing.T) {
	ref, err := parseNarinfo([]byte(sampleNarinfo))
	if err != nil {
		t.Fatal(err)
	}
	if ref.storePath != "/nix/store/1b9p3x1z5r7c2m8k4w6v0q9s3d5f7h1j-hermes-agent" {
		t.Errorf("storePath = %q", ref.storePath)
	}
	if ref.identity.narHash != "sha256:def" || ref.identity.narSize != "100" {
		t.Errorf("identity = %+v", ref.identity)
	}
	// References are sorted so that equal sets compare equal across platforms.
	want := "1b9p3x1z5r7c2m8k4w6v0q9s3d5f7h1j-dep-a 6v4d2c8b0ssfmzqk3p9lhrx5j7wn1dyg-dep-b"
	if ref.identity.references != want {
		t.Errorf("references = %q, want %q", ref.identity.references, want)
	}
}

func TestParseNarinfoRejectsMissingIdentity(t *testing.T) {
	data := strings.Replace(sampleNarinfo, "NarHash: sha256:def\n", "", 1)
	if _, err := parseNarinfo([]byte(data)); err == nil {
		t.Fatal("want error for narinfo without NarHash")
	}
}

func TestFlattenNarinfoRewritesURL(t *testing.T) {
	flat, name, err := flattenNarinfo([]byte(sampleNarinfo))
	if err != nil {
		t.Fatal(err)
	}
	if name != "abc123.nar.xz" {
		t.Errorf("nar name = %q", name)
	}
	got := string(flat)
	if !strings.Contains(got, "\nURL: abc123.nar.xz\n") {
		t.Errorf("flattened narinfo lacks flat URL:\n%s", got)
	}
	if strings.Contains(got, "nar/") {
		t.Errorf("flattened narinfo still points under nar/:\n%s", got)
	}
}

func TestFlattenNarinfoRejectsBadURL(t *testing.T) {
	tests := map[string]string{
		"nested path": strings.Replace(sampleNarinfo, "URL: nar/abc123.nar.xz", "URL: nar/x/abc.nar.xz", 1),
		"no nar/":     strings.Replace(sampleNarinfo, "URL: nar/abc123.nar.xz", "URL: abc123.nar.xz", 1),
		"empty name":  strings.Replace(sampleNarinfo, "URL: nar/abc123.nar.xz", "URL: nar/", 1),
	}
	for name, data := range tests {
		t.Run(name, func(t *testing.T) {
			if _, _, err := flattenNarinfo([]byte(data)); err == nil {
				t.Fatal("want error")
			}
		})
	}
}
