# This file defines a complete development environment for this project, using
# the Nix package manager. You can optionally use this instead of the usual
# `rustup` tools. If you want to do this, first install Nix. (You don't need
# NixOS for this; the Nix package manager runs on most platforms.)
#
#   https://nixos.org/download.html
#
# Then run `nix-shell` in this directory. It'll download any dependencies you
# need and then start a shell with those dependencies available. It won't
# change anything that's installed system-wide, and you don't need to use
# `sudo` when running `nix-shell`.
#
# For more advanced usage, try direnv and lorri:
#   https://direnv.net/
#   https://github.com/nix-community/lorri

{ sources ? import nix/sources.nix }:

# All dependencies are pinned to exact versions using niv, which manages
# `nix/sources.*`.
#   https://github.com/nmattia/niv
# To update dependencies to current versions, run `niv update`.

let
  pkgs = import sources.nixpkgs {
    overlays = [ (import sources.rust-overlay) ];
  };

  rust-nightly = pkgs.rust-bin.selectLatestNightlyWith (toolchain: toolchain.minimal);

  # Some tools require a nightly build of rustc. This function wraps those with
  # a shell script to put an appropriate version on their path.
  with-nightly = pkg: pkgs.runCommand "${pkg.name}-nightly" {
    nativeBuildInputs = [ pkgs.makeWrapper ];
  } ''
    for f in ${pkg}/bin/*; do
      makeWrapper "$f" $out/bin/"$(basename "$f")" \
        --prefix PATH : ${pkgs.lib.makeBinPath [ rust-nightly ]}
    done
  '';

in pkgs.mkShell {
  nativeBuildInputs = [
    pkgs.rust-bin.stable.latest.default
    (with-nightly pkgs.cargo-fuzz)

    # To update the versions of tools installed in this development
    # environment, run `niv update`.
    pkgs.niv
  ];
}
