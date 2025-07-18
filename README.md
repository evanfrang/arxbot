# arxbot
Evan Frangipane

## Introduction

The goal of this project is to create a simple RAG pipeline. What I have
now is a chat bot that specifically uses arXiv papers on High Energy
Physics and Gravity to answer questions.

## RAG

### Search

I wanted to use a hybrid search system which combines the best of
lexical search (exact matches) and vector search (semantic
i.e. meanings). I couldn’t get the vector search (Specter 2) to work
well on titles and abstracts so I am currently just using BM25 for
semantic search.

### LLM model

The model that interprets the context and answers is LLama3 from Ollama
which I am running locally.

## Example Output

Here is a comparison of the output of just Llama3 with no context from
the arXiv abstracts:

    Ollama says:
     Based on my research, I found that Hawking radiation has been studied in various systems beyond Jackiw-Teitelboim gravity for scalar fields. Some examples include:

    * Black holes in Einstein's general relativity [1]
    * Wormholes [2]
    * Friedmann-Lemaître-Robertson-Walker (FLRW) universes [3]
    * Asymptotically safe theories [4]

    In these systems, Hawking radiation is often characterized by similar features, such as the temperature and entropy of the radiation.

    References:

    [1] Hawking, S. W. (1974). Black hole explosions?. Nature, 248(5445), 30-31.

    [2] Morris, M. G., & Thorne, K. S. (1988). Wormholes, black holes, and time machines. Physical Review Letters, 61(13), 1476-1479.

    [3] Bardeen, J. M., Carter, B., & Hawking, S. W. (1973). The four laws of black hole mechanics. Communications in Mathematical Physics, 31(2), 161-170.

    [4] Reuter, F., & Wetterich, C. (1994). Asymptotic safety: A new approach to quantum gravity?. Nuclear Physics B, 392(1-2), 95-126.

    Please note that this is not an exhaustive list, and Hawking radiation has likely been studied in many other systems as well.

And now for the adding context using my RAG pipeline:

    Ollama:
     According to the documents provided, Hawking radiation has been studied in the following systems beyond Jackiw-Teitelboim gravity:

    1. Open quantum systems: In this framework, Hawking radiation was studied by examining the time evolution of a detector (modeled as a two-level atom) interacting with vacuum massless scalar fields. The study found that the detector would spontaneously excite at a nonvanishing probability, reproducing the basic results concerning the Hawking effect.
    2. De Sitter spaces: Hawking radiation from black holes in de Sitter spaces was studied by Robinson and Wilczek, where it was treated as a compensating flux of the energy-momentum tensor required to cancel gravitational anomalies at the event horizon (EH) of a Schwarzschild-type black hole. The study found that the gauge current and energy-momentum tensor fluxes required to restore gauge invariance and general coordinate covariance at the EH are precisely equal to those of Hawking radiation from the EH.
    3. Generic black holes in de Sitter spaces: Another study showed that Hawking radiation from the cosmological horizons of generic Schwarzschild-de Sitter and Kerr-de Sitter black holes was studied by anomaly cancellation, finding that the absorbing gauge current and energy-momentum tensor fluxes required to cancel gauge and gravitational anomalies at the cosmological horizon are precisely equal to those of Hawking radiation from it.

    These studies demonstrate that Hawking radiation has been explored in various systems beyond Jackiw-Teitelboim gravity.

## Conclusion

This isn’t better or worse but it is more specific when given context.
