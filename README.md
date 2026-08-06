# LLM_Viz

An interactive visual guide to the design choices inside modern language models.
Explore the forward pass as thirteen fixed positions, compare architectural
variants through lineage maps, inspect concept maps, and load Hugging Face model
configs to see the path a real model takes through the design space.

## Run locally

```bash
npm install
npm run dev
```

Open `http://localhost:5173/LLM_Viz/`.

## Validate and build

```bash
npm run check
npm run build
```

## Deploy

Push to `main` to deploy through GitHub Actions. The published site is
[dibalokechanda.github.io/LLM_Viz](https://dibalokechanda.github.io/LLM_Viz/).
