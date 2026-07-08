"""
Prepare demo dataset: embed texts, build HNSW index, compute 2D positions.

Run this script once to create the pre-loaded index for the frontend demo.
"""

import json
import numpy as np
from hnsw_instrumented import InstrumentedHNSW
from embedder import Embedder

# ──────────────────────────────────────────────────────────────────────
# Demo texts — diverse topics to show clustering behavior
# ──────────────────────────────────────────────────────────────────────

DEMO_TEXTS = [
    # Physics & Space (cluster)
    "Black holes form when massive stars collapse under their own gravity at the end of their life cycle",
    "The event horizon is the boundary around a black hole beyond which nothing can escape",
    "Neutron stars are incredibly dense remnants of supernova explosions",
    "Gravitational waves are ripples in spacetime caused by accelerating massive objects",
    "Dark matter makes up about 27% of the universe but cannot be directly observed",
    "The Big Bang theory describes the origin of the universe from a singularity 13.8 billion years ago",
    "Quantum entanglement allows particles to be correlated regardless of distance between them",
    "The speed of light in vacuum is approximately 299,792,458 meters per second",
    "Nuclear fusion powers stars by combining hydrogen atoms into helium at extreme temperatures",
    "The Milky Way galaxy contains between 100 and 400 billion stars",
    "Supernovae are powerful explosions that occur when stars exhaust their nuclear fuel",
    "The Hubble Space Telescope has captured images of galaxies billions of light years away",
    "Einstein's theory of general relativity describes gravity as curvature of spacetime",
    "Cosmic microwave background radiation is the afterglow of the Big Bang",
    "Pulsars are rapidly rotating neutron stars that emit beams of electromagnetic radiation",
    "The observable universe has a diameter of approximately 93 billion light years",
    "Antimatter is composed of antiparticles which have the same mass but opposite charge",
    "String theory proposes that fundamental particles are one-dimensional strings vibrating at different frequencies",
    "Dark energy is thought to be responsible for the accelerating expansion of the universe",
    "The James Webb Space Telescope observes the universe in infrared light",

    # Biology & Medicine (cluster)
    "DNA contains the genetic instructions for the development and functioning of living organisms",
    "CRISPR-Cas9 is a revolutionary gene editing tool that can precisely modify DNA sequences",
    "Mitochondria are the powerhouses of the cell that generate most of the cell's ATP energy",
    "The human genome contains approximately 3 billion base pairs of DNA",
    "Vaccines work by training the immune system to recognize and fight specific pathogens",
    "Antibiotics kill bacteria or stop their growth but are ineffective against viruses",
    "Photosynthesis converts sunlight carbon dioxide and water into glucose and oxygen in plants",
    "Evolution by natural selection drives the adaptation of species over generations",
    "The human brain contains approximately 86 billion neurons connected by trillions of synapses",
    "Stem cells have the unique ability to develop into many different types of specialized cells",
    "mRNA vaccines deliver genetic instructions for cells to produce harmless viral proteins",
    "The gut microbiome contains trillions of bacteria that influence digestion and immunity",
    "Cancer occurs when cells divide uncontrollably due to mutations in growth-regulating genes",
    "Insulin is a hormone produced by the pancreas that regulates blood sugar levels",
    "Enzymes are proteins that catalyze biochemical reactions by lowering activation energy",
    "The double helix structure of DNA was discovered by Watson and Crick in 1953",
    "Antibodies are Y-shaped proteins produced by the immune system to neutralize pathogens",
    "Neurons communicate through electrical impulses and chemical neurotransmitters",
    "Epigenetics studies how gene expression can be modified without changing the DNA sequence",
    "The circulatory system transports oxygen nutrients and waste products throughout the body",

    # Computer Science & AI (cluster)
    "Neural networks are computing systems inspired by biological neural networks in the brain",
    "Machine learning algorithms improve their performance on tasks through experience and data",
    "The transformer architecture revolutionized natural language processing with attention mechanisms",
    "GPT models are large language models trained to predict the next token in a sequence",
    "Gradient descent is an optimization algorithm that minimizes loss functions iteratively",
    "Convolutional neural networks excel at image recognition by detecting spatial patterns",
    "Reinforcement learning trains agents to make decisions by maximizing cumulative reward",
    "Binary search reduces the search space by half with each comparison in sorted data",
    "Hash tables provide average O(1) time complexity for insertion and lookup operations",
    "Recursion is a programming technique where a function calls itself to solve subproblems",
    "Big O notation describes the upper bound of an algorithm's time or space complexity",
    "Graph algorithms like Dijkstra's find shortest paths between nodes in weighted networks",
    "Distributed systems coordinate multiple computers to work together as a single system",
    "Docker containers package applications with their dependencies for consistent deployment",
    "Kubernetes orchestrates containerized applications across clusters of machines",
    "TCP/IP is the fundamental protocol suite that enables communication on the internet",
    "Encryption transforms data into unreadable form that can only be decoded with a key",
    "Blockchain is a distributed ledger technology that records transactions in linked blocks",
    "Version control systems like Git track changes to code over time enabling collaboration",
    "API design involves creating interfaces that allow different software systems to communicate",

    # History (cluster)
    "The Roman Empire lasted for over a thousand years and shaped Western civilization",
    "The Industrial Revolution transformed manufacturing through mechanization in the 18th century",
    "World War II was the deadliest conflict in human history with over 70 million casualties",
    "The French Revolution of 1789 overthrew the monarchy and established democratic ideals",
    "Ancient Egypt built the pyramids as tombs for pharaohs over 4500 years ago",
    "The Renaissance was a cultural movement that began in Italy in the 14th century",
    "The Cold War was a geopolitical rivalry between the United States and Soviet Union",
    "The printing press invented by Gutenberg around 1440 revolutionized the spread of knowledge",
    "The fall of the Berlin Wall in 1989 symbolized the end of the Cold War era",
    "Alexander the Great conquered much of the known world by the age of thirty",
    "The Silk Road was an ancient trade network connecting East Asia to the Mediterranean",
    "The American Revolution established the United States as an independent nation in 1776",
    "The Ottoman Empire was a powerful state that lasted from 1299 to 1922",
    "The Scientific Revolution of the 16th and 17th centuries transformed understanding of nature",
    "Mahatma Gandhi led India's nonviolent independence movement against British colonial rule",
    "The Space Race between the US and USSR culminated in the Apollo 11 moon landing in 1969",
    "The abolition of slavery was a major social reform movement of the 19th century",
    "The Black Death killed approximately one third of Europe's population in the 14th century",
    "Ancient Greece developed democracy philosophy and laid foundations for Western thought",
    "The Chinese Cultural Revolution was a sociopolitical movement launched by Mao Zedong in 1966",

    # Geography & Environment (cluster)
    "The Amazon Rainforest produces approximately 20% of the world's oxygen",
    "Climate change is causing global temperatures to rise due to greenhouse gas emissions",
    "The Pacific Ocean is the largest and deepest ocean covering more than 30% of Earth's surface",
    "Coral reefs support 25% of all marine species despite covering less than 1% of the ocean floor",
    "The Sahara Desert is the largest hot desert in the world spanning 9.2 million square kilometers",
    "Plate tectonics explains how Earth's lithosphere is divided into moving plates that cause earthquakes",
    "The Arctic ice cap is shrinking at a rate of approximately 13% per decade due to global warming",
    "Mount Everest is the highest point on Earth at 8,849 meters above sea level",
    "Deforestation destroys approximately 10 million hectares of forest every year",
    "The Great Barrier Reef is the world's largest coral reef system visible from space",
    "Volcanic eruptions release gases and ash that can affect global climate patterns",
    "Ocean acidification threatens marine ecosystems as CO2 dissolves in seawater",
    "The Mariana Trench is the deepest point in the ocean at nearly 11,000 meters deep",
    "Renewable energy sources include solar wind hydroelectric and geothermal power",
    "Biodiversity loss threatens ecosystem stability and human food security",
    "El Niño is a climate pattern that warms Pacific waters and affects weather worldwide",
    "The ozone layer protects Earth from harmful ultraviolet radiation from the sun",
    "Glaciers contain about 69% of the world's fresh water supply",
    "Tsunamis are caused by underwater earthquakes and can travel across entire oceans",
    "Permafrost in Arctic regions stores massive amounts of carbon that could accelerate warming if thawed",

    # Economics & Business (cluster)
    "Supply and demand determines prices in a free market economy",
    "Inflation erodes purchasing power when the general price level rises over time",
    "The stock market allows companies to raise capital by selling shares to investors",
    "GDP measures the total value of goods and services produced by a country in a year",
    "Venture capital firms invest in early-stage startups in exchange for equity ownership",
    "Central banks use interest rates to control inflation and stimulate economic growth",
    "Cryptocurrency operates on decentralized networks without traditional banking intermediaries",
    "Monopolies occur when a single company dominates a market reducing competition",
    "International trade allows countries to specialize in goods they produce most efficiently",
    "Recessions are periods of economic decline characterized by falling GDP and rising unemployment",
    "The gig economy relies on short-term contracts and freelance work rather than permanent jobs",
    "Compound interest causes investments to grow exponentially over long time periods",
    "Market bubbles form when asset prices far exceed their fundamental value",
    "Tax policy affects wealth distribution and government revenue for public services",
    "Startups often follow a lean methodology of rapid iteration based on customer feedback",

    # Food & Cooking (cluster — different from everything else)
    "Bread baking requires flour water yeast and salt combined with proper fermentation time",
    "Sushi originated in Japan as a method of preserving fish in fermented rice",
    "Chocolate is made from cacao beans that are fermented dried roasted and ground",
    "Italian pasta comes in over 300 different shapes each designed for specific sauces",
    "Fermentation transforms food through beneficial bacteria producing yogurt kimchi and beer",
    "The Maillard reaction creates flavour and browning when proteins and sugars are heated",
    "Olive oil is extracted from olives and is a staple of Mediterranean cuisine",
    "Spices like turmeric cumin and cinnamon have been traded for thousands of years",
    "Coffee beans are actually seeds from berries of the Coffea plant",
    "Umami is the fifth basic taste discovered in 1908 found in foods like mushrooms and parmesan",

    # Music & Arts (cluster)
    "Classical music evolved through periods including Baroque Classical Romantic and Modern",
    "The piano was invented around 1700 and has 88 keys spanning over seven octaves",
    "Jazz originated in New Orleans blending African rhythms with European harmony",
    "The Beatles transformed popular music in the 1960s with innovative recording techniques",
    "Oil painting became the dominant medium for fine art during the Renaissance period",
    "Hip hop emerged from African American communities in the Bronx during the 1970s",
    "Symphony orchestras typically contain between 70 and 100 musicians across four sections",
    "Abstract art abandons realistic representation in favor of shapes colors and textures",
    "The guitar is one of the most popular instruments with origins dating back 4000 years",
    "Film editing uses techniques like montage and cross-cutting to create narrative meaning",

    # Sports (cluster)
    "The Olympic Games originated in ancient Greece and were revived in 1896 in Athens",
    "Soccer is the most popular sport in the world with over 4 billion fans globally",
    "Basketball was invented by James Naismith in 1891 using a peach basket as the goal",
    "Marathon running covers 42.195 kilometers and commemorates a legendary Greek messenger",
    "Tennis evolved from a French handball game and is now played on four different surfaces",
    "Cricket is played by over 2.5 billion fans primarily in Commonwealth nations",
    "Formula One racing combines engineering excellence with driver skill at speeds over 300 km/h",
    "Swimming has been an Olympic sport since the first modern games in 1896",
    "The FIFA World Cup is the most watched sporting event attracting billions of viewers",
    "Rock climbing requires strength flexibility and problem-solving to ascend vertical surfaces",

    # Psychology & Philosophy (cluster)
    "Cognitive behavioral therapy helps people change negative thinking patterns and behaviors",
    "The unconscious mind influences behavior without conscious awareness according to Freud",
    "Stoicism teaches that virtue and wisdom come from controlling one's reactions to external events",
    "Memory formation involves encoding storage and retrieval processes in the hippocampus",
    "Existentialism emphasizes individual freedom responsibility and the search for meaning",
    "The placebo effect demonstrates how belief alone can produce measurable physical changes",
    "Maslow's hierarchy of needs describes human motivation from basic survival to self-actualization",
    "Meditation has been shown to reduce stress and improve attention and emotional regulation",
    "Confirmation bias leads people to favor information that confirms their existing beliefs",
    "The trolley problem is a thought experiment exploring the ethics of sacrificing one to save many",
]

print(f"Dataset: {len(DEMO_TEXTS)} texts across ~9 topic clusters")


# ──────────────────────────────────────────────────────────────────────
# Embed all texts
# ──────────────────────────────────────────────────────────────────────

print("Loading embedder...")
embedder = Embedder()
print(f"Embedding {len(DEMO_TEXTS)} texts (dim={embedder.dim})...")
vectors = embedder.embed_batch(DEMO_TEXTS)
print(f"  Shape: {vectors.shape}")


# ──────────────────────────────────────────────────────────────────────
# Build HNSW index
# ──────────────────────────────────────────────────────────────────────

print("\nBuilding HNSW index...")
index = InstrumentedHNSW(M=16, ef_construction=200, metric="l2", seed=42)
for v in vectors:
    index.insert(v)
print(f"  {index}")


# ──────────────────────────────────────────────────────────────────────
# Compute 2D positions using PCA (fast, no extra dependencies)
# ──────────────────────────────────────────────────────────────────────

print("\nComputing 2D projection (PCA)...")
# Center the data
mean = vectors.mean(axis=0)
centered = vectors - mean
# SVD for top 2 components
U, S, Vt = np.linalg.svd(centered, full_matrices=False)
positions = U[:, :2] * S[:2]  # project onto first 2 principal components

# Normalize to [0, 1] range for frontend
pos_min = positions.min(axis=0)
pos_max = positions.max(axis=0)
positions_normalized = (positions - pos_min) / (pos_max - pos_min + 1e-10)

print(f"  Positions shape: {positions_normalized.shape}")
print(f"  Range: [{positions_normalized.min():.2f}, {positions_normalized.max():.2f}]")


# ──────────────────────────────────────────────────────────────────────
# Save everything
# ──────────────────────────────────────────────────────────────────────

print("\nSaving...")
index.save("saved_index")
np.save("positions.npy", positions_normalized)
with open("texts.json", "w") as f:
    json.dump(DEMO_TEXTS, f)

print(f"\n✓ Done! Files created:")
print(f"  saved_index/     — HNSW index (vectors + graph)")
print(f"  positions.npy    — 2D positions for visualization")
print(f"  texts.json       — original text strings")
print(f"\nRun 'python server.py' to start the API.")
