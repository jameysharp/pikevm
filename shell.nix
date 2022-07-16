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
in pkgs.mkShell {
  nativeBuildInputs = [
    pkgs.rust-bin.stable.latest.default

    # To update the versions of tools installed in this development
    # environment, run `niv update`.
    pkgs.niv
  ];
}
