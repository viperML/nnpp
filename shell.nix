with import <nixpkgs> { };
mkShell.override
  {
    stdenv = llvmPackages.stdenv;
  }
  {
    packages = [
      cmake
      ninja
      neocmakelsp
      clang-tools
      (callPackage ./nix/structopt.nix { })
    ];

    env = {
      CMAKE_EXPORT_COMPILE_COMMANDS = true;
      CMAKE_GENERATOR = "Ninja";
    };
  }
