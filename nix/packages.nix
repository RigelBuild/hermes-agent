# nix/packages.nix — Hermes Agent package built with uv2nix
{ inputs, ... }:
{
  perSystem =
    {
      pkgs,
      lib,
      inputs',
      ...
    }:
    let

      sandbox = pkgs.callPackage ./sandbox.nix { };

      # pm/lock.json translated into derivations — one per tool package
      # with an artifact for this system. The lockfile carries resolved
      # urls + hashes, so nix consumes it as pure data. callPackage adds
      # .override/.overrideDerivation to the returned SET; keep only the
      # real derivations.
      pmPackages = lib.filterAttrs (_: lib.isDerivation) (
        pkgs.callPackage ./pm-packages.nix { }
      );

      dirtyRevision = inputs.self.dirtyRev or null;
      # inputs.self.sourceInfo is a string (store path) in current Nix, not a
      # record. The metadata fields are available as top-level attributes on
      # inputs.self directly. A flake URL's requested ref is intentionally not
      # exposed as self.ref after Nix resolves it to an immutable source, so
      # branch is normally null for remote flakes. Preserve null rather than
      # inventing a sentinel that could be a real branch name.
      rev =
        inputs.self.rev or (if dirtyRevision != null then builtins.substring 0 40 dirtyRevision else null);
      rawRef = inputs.self.ref or null;
      branch = if rawRef != null then builtins.replaceStrings [ "refs/heads/" ] [ "" ] rawRef else null;
      dirty = dirtyRevision != null;
      lastModified = inputs.self.lastModified or null;
      minimal = pkgs.callPackage ./hermes-agent.nix {
        inherit (inputs) uv2nix pyproject-nix pyproject-build-systems;
        # Passed explicitly so `override (prev: ...)` sees it: a function form that appends to
        # `prev.extraDependencyGroups` worked before the shipped groups moved to
        # defaultDependencyGroups, and an argument default never lands in `prev`.
        extraDependencyGroups = [ ];
        inherit pmPackages;
        inherit
          rev
          branch
          dirty
          lastModified
          ;
      };

      # All platform-portable optional integrations pre-built. These are the
      # variant's own groups, not the user's extras, so an
      # `.override { extraDependencyGroups = ...; }` adds to them.
      full = minimal.override {
        defaultDependencyGroups = [
          "anthropic"
          "azure-identity"
          "bedrock"
          "daytona"
          "dingtalk"
          "edge-tts"
          "exa"
          "fal"
          "feishu"
          "firecrawl"
          "messaging"
          "modal"
          "parallel-web"
          "tts-premium"
          "vercel"
          "voice"
        ]
        # matrix is Linux-only (oqs/liboqs lacks aarch64-darwin wheels).
        ++ lib.optionals pkgs.stdenv.hostPlatform.isLinux [ "matrix" ];
      };
    in
    {
      packages = {
        node-gyp =
          (pkgs.callPackage ./lib.nix {
            inherit pmPackages;
          }).node-gyp;
        default = full;

        # The release output is kept out of `packages` (see legacyPackages
        # below): a package must build on every system it is listed for.
        inherit sandbox;

        inherit minimal;

        # Ships discord.py + python-telegram-bot + slack-sdk so a plain
        # `nix profile install .#messaging` connects to Discord/Telegram/Slack
        # on first run — lazy-install can't write to the read-only /nix/store.
        messaging = minimal.override {
          defaultDependencyGroups = [ "messaging" ];
        };

        tui = full.hermesTui;
        web = full.hermesWeb;
        webapp = full.hermesWebapp;
        desktop = full.hermesDesktop;

        # The fixed-output pnpm store every frontend installs from; its hash
        # lives in nix/pnpm-deps-hash/<system> (`nix run .#update-pnpm-lockfile`).
        pnpm-deps = full.hermesNpmLib.pnpmDeps;
        update-pnpm-lockfile = full.hermesNpmLib.updatePnpmLockfile;
      }
      # Every pm lockfile tool as its own installable derivation:
      # `nix build .#pm-ripgrep`, `nix build .#pm-gh`, ...
      // lib.mapAttrs' (name: drv: lib.nameValuePair "pm-${name}" drv) pmPackages;

      # Release closure from the matching GitHub Release. Not a package, and
      # `nix flake check` never forces it. The pin is checked before
      # builtins.fetchClosure is referenced, because that builtin exists only
      # with the fetch-closure experimental feature.
      legacyPackages.release =
        let
          system = pkgs.stdenv.hostPlatform.system;
          pinFile = ./release-closure + "/${system}.json";
        in
        if !builtins.pathExists pinFile then
          throw "No release closure pin for ${system}; expected nix/release-closure/${system}.json"
        else
          let
            pin = builtins.fromJSON (builtins.readFile pinFile);
          in
          if !(builtins.isAttrs pin && builtins.attrNames pin == [ "storePath" "tag" ]) then
            throw "nix/release-closure/${system}.json must contain exactly two keys, tag and storePath"
          else if !(builtins.isString pin.tag && builtins.isString pin.storePath) then
            throw "nix/release-closure/${system}.json fields tag and storePath must be strings"
          else if !(builtins.match "v[A-Za-z0-9._+-]*-rigel\\.[A-Za-z0-9._+-]+" pin.tag != null) then
            throw "nix/release-closure/${system}.json tag must match v*-rigel.*"
          else
            pkgs.runCommandLocal "hermes-agent-release"
              {
                closure = builtins.fetchClosure {
                  fromStore = "https://github.com/RigelBuild/hermes-agent/releases/download/${pin.tag}";
                  fromPath = pin.storePath;
                  inputAddressed = true;
                };
                meta.mainProgram = "hermes";
              }
              ''
                mkdir -p "$out/bin"
                ln -s "$closure/bin/hermes" "$out/bin/hermes"
              '';
    };
}
