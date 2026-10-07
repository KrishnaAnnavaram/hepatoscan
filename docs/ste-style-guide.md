# The writing standard: ASD-STE100 Simplified Technical English

The README of hepatoscan and this file obey these rules. Section 3 gives the project vocabulary: the technical names and the technical verbs of hepatoscan.

## 1. The writing rules

### Words

1. Use one word for one meaning, and one meaning for one word. Do not use synonyms for variety.
2. Use a word only as one part of speech. For example, `test` is a noun or a verb, `check` is a verb.
3. Do not use phrasal verbs (`set up`, `carry out`, `find out`, `pick up`, `look up`, `come up with`).
   Use one verb: `prepare`, `do`, `find`, `get`, `make`.
4. Do not use an `-ing` form as a noun or an adjective (`the running job`, `after indexing`).
   Exception: a technical name, a file name, a command or a status value.
5. Do not use contractions (`don't`, `it's`, `can't`). Do not use slang or idioms
   (`out of the box`, `under the hood`, `at a glance`, `gotcha`, `bells and whistles`).
6. Do not use `and/or`. Write `A, B or both`.
7. Do not use `should`, `could`, `would` or `may` for instructions. Use `must` for a rule, the
   imperative for a step and `can` for a possibility.
8. Keep the articles `a`, `an` and `the` in sentences.
9. Do not make a noun cluster of more than three words. A technical name is one word.

### Sentences

1. A procedural sentence (an instruction) has a maximum of **20 words**.
2. A descriptive sentence has a maximum of **25 words**.
3. Write one instruction in one sentence.
4. Use the imperative for an instruction: `Run the tests.` Not `The tests should be run.`
5. Use the active voice. Use the passive voice only when the agent of the action is not important.
6. Use only the simple present, the simple past and the simple future.
7. Put a condition before the instruction: `If the index is stale, build it again.`
8. Do not use semicolons in sentences. Write two sentences.

### Paragraphs, notes and warnings

1. A paragraph has one topic and a maximum of **6 sentences**. Start with the topic sentence.
2. A warning or a caution starts with a clear command. Then it gives the reason.
3. A note gives information. It does not give an instruction.
4. Use a vertical list for a sequence or a set of conditions. Each item of a numbered procedure is one step.

### Tables, headings and diagrams

1. A table cell can be a short phrase. If a cell has a sentence, the sentence obeys the rules.
2. A heading is a noun phrase (`The cost model`) or an imperative (`Run the demo`).
   Do not start a heading with an `-ing` form.
3. A diagram label is a short phrase. Use the same terms as the text.

### What STE does not change

Code, commands, file names, paths, field names, environment variables, status values, enum values,
product names and URLs stay exactly as they are. They are technical names. Put them in backticks.

## 2. General words to replace

| Do not use | Use |
|---|---|
| utilize, leverage | use |
| in order to | to |
| set up | prepare, install, configure |
| carry out, perform | do |
| make sure, ensure | make sure (allowed), or `check that` |
| a lot of, lots of | many, much |
| e.g., i.e. | for example, that is |
| should (instruction) | must (rule) / imperative (step) |
| might, may (possibility) | can |
| very, really, just, simply, easily | (delete) |
| seamless, robust, powerful, blazing | (delete or give a measured fact) |

## 3. Project vocabulary

These terms have one meaning in the hepatoscan documentation. Code names are in backticks.

### 3.1 Technical names (nouns)

| Term | Meaning | Do not use |
|---|---|---|
| **volume** | One 3-D CT image in Hounsfield units, with its voxel spacing. | scan (for the data), image stack, study |
| **case** | One volume with its mask and its `case_id`. | patient, sample, subject |
| **mask** | A 3-D array of labels with the same shape as the volume. | segmentation (for the array), label map |
| **label** | One mask value: `0` background, `1` liver, `2` tumor. | class id, category |
| **liver region** | All voxels with label 1 or 2. | liver mask, organ |
| **tumor region** | All voxels with label 2. | lesion mask, tumour |
| **lesion** | One connected component of the tumor region. | tumor (for one component), nodule, spot |
| **phantom** | A synthetic volume with an exact mask from `make_phantom`. | fake scan, dummy CT |
| **window** | A Hounsfield range (level and width) that is scaled to [0, 1]. | contrast setting, filter |
| **channel** | One windowed copy of the volume. | band, layer |
| **2.5-D input** | The channels of one slice and its neighbour slices. | stack, multi-slice |
| **segmenter** | A component that returns a mask for a volume: `voxel-gbm` or `attention-unet-2.5d`. | model (alone), predictor |
| **baseline** | The voxel gradient-boosting segmenter `voxel-gbm`. | classic model, simple model |
| **checkpoint** | A `.pt` file with the U-Net weights, the U-Net config and the preprocessing config. | weights file, model file |
| **split** | The disjoint train, validation and test sets of case ids. | fold, partition |
| **summary** | The measured liver volume, lesion count and lesion volume of one mask. | report, findings, result |
| **upload** | The bytes that a client sends to `/segment`. | file (alone), attachment |
| **audit log** | The append-only JSON-lines record of service events. | log (alone), history |
| **assistant** | The question-answer component with retrieval and safety rules. | chatbot, bot, AI |
| **knowledge base** | The Markdown documents in `assistant/kb`. | corpus, documents (alone) |
| **passage** | One paragraph of a knowledge-base document. | chunk, snippet |
| **citation** | A marker `[n]` that points to a passage in the prompt. | reference, footnote |
| **red flag** | A phrase that can describe an emergency. | danger word, alarm |
| **escalation** | The fixed message that tells the user to get urgent care. | alert, warning (for this message) |
| **refusal** | The fixed non-diagnostic message for a diagnosis, prognosis or dose request. | rejection, block |
| **session** | The server-side history of one conversation, with a cap and a time limit. | chat history, memory |
| **token** | The bearer secret in `HEPATOSCAN_API_TOKEN`. | password, key (for this secret) |

### 3.2 Technical verbs

| Verb | Meaning |
|---|---|
| **segment** | Make a mask for a volume with a segmenter. |
| **preprocess** | Resample a volume and make its channels with `preprocess_volume`. |
| **resample** | Change the voxel spacing (linear for images, nearest neighbour for masks). |
| **validate** | Check an upload, a volume or a mask against its schema. |
| **evaluate** | Measure Dice, HD95 and lesion detection for each test volume. |
| **retrieve** | Get the passages for a question with BM25. |
| **cite** | Put a marker `[n]` for a passage in an answer. |
| **escalate** | Give the escalation message and do not call the language model. |
| **refuse** | Give the refusal message before the general information. |
| **audit** | Write one record to the audit log. |
