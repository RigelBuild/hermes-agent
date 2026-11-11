// Command release-cache builds and pins the flat Nix binary cache that
// hermes-agent publishes as GitHub Release assets.
//
//	release-cache assemble -out DIR [-upstream URL] PLATFORM_DIR...
//	release-cache write-pin -system SYSTEM -tag TAG -path-info FILE -out FILE
package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
)

const usage = `usage:
  release-cache assemble -out DIR [-upstream URL] PLATFORM_DIR...
  release-cache write-pin -system SYSTEM -tag TAG -path-info FILE -out FILE
`

func main() {
	if err := run(os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "release-cache:", err)
		os.Exit(1)
	}
}

func run(args []string, stdout io.Writer) error {
	if len(args) == 0 {
		return errors.New(usage)
	}
	switch args[0] {
	case "assemble":
		return runAssemble(args[1:], stdout)
	case "write-pin":
		return runWritePin(args[1:])
	default:
		return fmt.Errorf("unknown command %q\n%s", args[0], usage)
	}
}

func runAssemble(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("assemble", flag.ContinueOnError)
	out := fs.String("out", "", "empty directory to write the flat release assets into")
	upstream := fs.String("upstream", "", "skip paths this cache serves, e.g. https://cache.nixos.org")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *out == "" || fs.NArg() == 0 {
		return errors.New("assemble needs -out and at least one platform directory")
	}
	cfg := assembleConfig{platformDirs: fs.Args(), outDir: *out, maxAssets: maxReleaseAssets}
	if *upstream != "" {
		cfg.upstream = newHTTPUpstream(*upstream)
	}
	n, err := assemble(cfg)
	if err != nil {
		return err
	}
	fmt.Fprintf(stdout, "wrote %d release assets to %s\n", n, *out)
	return nil
}

func runWritePin(args []string) error {
	fs := flag.NewFlagSet("write-pin", flag.ContinueOnError)
	system := fs.String("system", "", "Nix system, e.g. x86_64-linux")
	tag := fs.String("tag", "", "release tag the closure is published under")
	pathInfo := fs.String("path-info", "", "nix path-info --json output for the built closure")
	out := fs.String("out", "", "pin file to write, e.g. nix/release-closure/x86_64-linux.json")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *system == "" || *tag == "" || *pathInfo == "" || *out == "" {
		return errors.New("write-pin needs -system, -tag, -path-info and -out")
	}
	return writePin(*system, *tag, *pathInfo, *out)
}
