package main

import (
	"bytes"
	"errors"
	"fmt"
	"slices"
	"strings"
)

// narinfo holds the fields the release tool reads from one narinfo file.
type narinfo struct {
	storePath string
	identity  narIdentity
}

// narIdentity is what must match when two platforms publish the same store
// path. Deriver and Sig may differ between platforms.
type narIdentity struct {
	narHash    string
	narSize    string
	references string
}

func parseNarinfo(data []byte) (narinfo, error) {
	var ref narinfo
	var refs []string
	for _, line := range bytes.Split(data, []byte("\n")) {
		key, value, ok := strings.Cut(string(line), ":")
		if !ok {
			continue
		}
		value = strings.TrimSpace(value)
		switch key {
		case "StorePath":
			ref.storePath = value
		case "NarHash":
			ref.identity.narHash = value
		case "NarSize":
			ref.identity.narSize = value
		case "References":
			refs = strings.Fields(value)
		}
	}
	if ref.storePath == "" || ref.identity.narHash == "" || ref.identity.narSize == "" {
		return narinfo{}, errors.New("narinfo lacks StorePath, NarHash or NarSize")
	}
	slices.Sort(refs)
	ref.identity.references = strings.Join(refs, " ")
	return ref, nil
}

// flattenNarinfo rewrites the URL line from nar/<file> to <file>, so the
// narinfo resolves against a release root that holds every asset flat. It
// returns the rewritten narinfo and the bare NAR file name.
func flattenNarinfo(data []byte) ([]byte, string, error) {
	lines := bytes.Split(data, []byte("\n"))
	name := ""
	found := 0
	for i, line := range lines {
		rest, ok := bytes.CutPrefix(line, []byte("URL: nar/"))
		if !ok {
			continue
		}
		if len(rest) == 0 || bytes.Contains(rest, []byte("/")) {
			return nil, "", fmt.Errorf("narinfo URL nar/%s is not a single file name", rest)
		}
		lines[i] = append([]byte("URL: "), rest...)
		name = string(rest)
		found++
	}
	if found != 1 {
		return nil, "", fmt.Errorf("want one URL line under nar/, found %d", found)
	}
	return bytes.Join(lines, []byte("\n")), name, nil
}
