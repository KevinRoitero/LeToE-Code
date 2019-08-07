FILE DESCRIPTION:
	- REPRO_SNC implements Soboroff, Nicholas and Cahan's method. It is based on the mean of relevant documents per topic and its standard deviation for a specific collection; these values have been calculated in a separate script (see below) and they are only read here. The artificial qrels are created sampling the documents contained in the real qrels according to a normal distribution built using the previously computed mean and standard deviation. These artificial qrels replace the real ones in the evaluation process using trec_eval, whose output formes the approximated MAP values. For each collection, the process is repeated 20 times to reduce randomness.
	- REPRO_SPO implements Spoerri's method. It is based on the percentage of documents retrieved only by a system and the percentage of documents that are retrieved by every other system in a chunk. A chunk is a set of system, in SPO's case a set of five systems build with these rules (implemented in "Split_Systems_Trials.ipynb" notebook):
    - no system must appear more than once in a chunk
    - no system must appear more than five times overall, considering all of the chunks
    - every chunk must be different (= contain different systems) from the others
The chunks have been computed in a separated script and they are only read and loaded in memory here. The method produces five different values for of the three each variants (SINGLE, ALLFIVE, SINGLE - ALLFIVE) per system and these values are then averaged to get a single value of each of the three variants. The final values aren't an approximated MAP but values proportional to the real MAP values.
	- REPRO_WUC implements Wu and Crestani's method. It is based on the count of documents retrieved by a system (original documents) that are retrieved also by other systems (reference documents), given a specific topic. This computation is carried on in five different variants:
    - BASIC sums the occurrences of the original documents in the results of all the other retrieval systems
    - V1 assigns different weights to the reference documents based on their ranking position (the weight function is 1501 - rank)
    - V2 assigns different weights to the original documents based on their raking position (the weight function is the Zeta function)
    - V3 assigns different weights to both the original documents and the reference documents based on their raking position (the weight function is 1501 - rank for the original documents and the Zeta function for the reference documents)
    - V4 assigns different weights to the reference documents based on their normalized relevance score (the weight function is the Zeta function)
The values of each document, given a topic, are then summed for each system and give a value for each system and each topic. These values aren't an approximation of MAP but values proportional to the real MAP values.
	- compute_mu_sigma computes these two parameters for both the "best" approaches: pool made from the top 100 documents retrieved from each system on each topic (removing the duplicates) and original qrels. For each topic, the script computes the mean of relevant documents in that topic and then averages this mean over all topics. For the parameter sigma, it computes the standard deviation of the mean computed for each topic.

contacts:
Marco Passon -- passon.marco@spes.uniud.it
Kevin Roitero -- roitero.kevin@spes.uniud.it