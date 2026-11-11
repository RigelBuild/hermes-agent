package main

import (
	"fmt"
	"net/http"
	"strings"
	"time"
)

// httpUpstream asks a binary cache whether it serves <hash>.narinfo.
type httpUpstream struct {
	base   string
	client *http.Client
}

func newHTTPUpstream(base string) httpUpstream {
	return httpUpstream{
		base:   strings.TrimSuffix(base, "/"),
		client: &http.Client{Timeout: 30 * time.Second},
	}
}

func (u httpUpstream) has(hash string) (bool, error) {
	req, err := http.NewRequest(http.MethodHead, u.base+"/"+hash+".narinfo", nil)
	if err != nil {
		return false, err
	}
	resp, err := u.client.Do(req)
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()
	switch resp.StatusCode {
	case http.StatusOK:
		return true, nil
	case http.StatusNotFound:
		return false, nil
	default:
		return false, fmt.Errorf("%s: HTTP %d", req.URL, resp.StatusCode)
	}
}
