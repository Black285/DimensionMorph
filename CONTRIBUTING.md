# Contributing to DimensionMorph

Contributions are welcome — bug fixes, new coordinate systems, new inputs,
documentation, tests.

## How to contribute

1. Fork the repository and create a branch from `main`.
2. Make your change. Keep it focused: one feature or fix per pull request.
3. Run the tests and make sure they pass:

   ```bash
   python -m pytest tests -q
   ```

4. Add tests for new behaviour. A new transform should be checked against its
   closed form, the way the existing ones in `tests/test_transforms.py` are.
5. Open a pull request describing what changed and why.

## Contributor License Agreement

Before your first pull request can be merged you need to sign the
[Contributor License Agreement](CLA.md). A bot will comment on your pull request
with instructions; signing takes one comment and covers all future
contributions.

In short: you keep the copyright to your work, but you give the maintainer
permission to use and distribute it — including under licences other than the
GPL-3.0. Read [CLA.md](CLA.md) for the full terms.

## Licence

DimensionMorph is licensed under the [GNU General Public License v3.0](LICENSE).
