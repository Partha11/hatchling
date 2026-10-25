# Contributing

Contributions are welcome, whether it is a bug fix, a new experiment or a correction to these pages.

## Reporting a problem

Open an [issue](https://github.com/Partha11/snake_rl_agent/issues) with your operating system, how you ran the game or the script, and what happened. For agent problems, the output of `play.py` or `diagnose.py` helps a lot.

## Making a change

1. Fork the repository.
2. Create a branch for your change, for example `git checkout -b feature/my-feature`.
3. Commit your changes with a clear message.
4. Push the branch and open a pull request.

Keep the engine independent of the game frontend and the Python code. It should only know the rules. If you change the rules or rewards in `include/snake/engine.hpp`, say so clearly in the pull request, because existing trained models were trained under the current values.

## Experiments

If your change affects how the agent learns or plays, include a 200 game [diagnostic](agent/diagnostics.md) of your model next to one of the released model. Results from single games are not enough to compare models.

Experiments that did not work are welcome too. Some of the most useful parts of this project came from failed attempts, and [Dead Ends](story/10-what-did-not-work.md) is the place for them.

## Documentation

These pages are built with MkDocs and the shadcn theme. To preview them locally, install the documentation dependencies and start the development server:

```bash
pip install pip-tools
pip-compile requirements-docs.in
pip install -r requirements-docs.txt
mkdocs serve
```

The site is then available at `http://127.0.0.1:8000`, and it reloads whenever you save a page.
