{
  description = "Python Flake";

  inputs = {
    nixpkgs.url = "https://github.com/NixOS/nixpkgs/archive/c407032be28ca2236f45c49cfb2b8b3885294f7f.tar.gz";
  };

  outputs =
    { self, nixpkgs, ... }@inputs:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      python = pkgs.python310;
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          python
          python.pkgs.numpy
          python.pkgs.scipy
          python.pkgs.venvShellHook
        ];

        venvDir = "./venv/";
        postShellHook = "pip install -q --upgrade pip && pip install -qr  requirements.txt";
      };
    };
}
