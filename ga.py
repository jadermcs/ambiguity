import operator
import random
import numpy as np
from typing import Tuple

import spacy
from deap import algorithms, base, creator, tools, gp
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
)

# --- 1. SETUP ---
# Load the spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading 'en_core_web_sm' model...")
    from spacy.cli import download

    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# --- 2. ENHANCED DATASET ---
# Expanded dataset with more diverse examples
SENTENCE_DATASET = [
    # Ambiguous sentences
    ("I saw the man with the telescope", True),  # PP attachment
    ("The visiting relatives can be boring", True),  # Gerund/present participle
    ("He saw that bat", True),  # Lexical ambiguity (animal vs. sports equipment)
    ("The man on the hill with a telescope saw a boy", True),  # Multiple PP attachment
    ("The horse raced past the barn fell", True),  # Garden path sentence
    ("Flying planes can be dangerous", True),  # Gerund vs. present participle
    ("The chicken is ready to eat", True),  # Infinitive clause ambiguity
    ("Time flies like an arrow", True),  # Multiple syntactic parses
    ("They are hunting dogs", True),  # Compound noun vs. progressive
    ("I saw her duck", True),  # Verb vs. noun ambiguity
    ("The professor talked to the student with the book", True),  # PP attachment
    ("Mary hit the man with the stick", True),  # Instrument vs. modifier
    ("We need more intelligent students", True),  # Scope ambiguity
    ("Every man loves a woman", True),  # Quantifier scope
    ("The shooting of the hunters was terrible", True),  # Subjective/objective genitive
    # Unambiguous sentences
    ("The cat sat on the mat", False),
    ("My dog eats food from a bowl", False),
    ("The sun rises in the east", False),
    ("She reads a new book every week", False),
    ("The clear blue sky is beautiful", False),
    ("John went to the store yesterday", False),
    ("The children played in the park", False),
    ("Coffee tastes bitter without sugar", False),
    ("Rain falls from the clouds", False),
    ("Birds fly south in winter", False),
    ("The teacher explained the lesson clearly", False),
    ("Students study hard for exams", False),
    ("Fresh flowers smell wonderful", False),
    ("Cars drive on roads", False),
    ("People walk with their feet", False),
]

# Pre-process the dataset with spaCy
PARSED_DATASET = [(nlp(sentence), label) for sentence, label in SENTENCE_DATASET]

# --- 3. ENHANCED FEATURE EXTRACTION FUNCTIONS ---


class LinguisticFeatures:
    """Class containing advanced linguistic feature extraction methods."""

    @staticmethod
    def count_pos_tag(doc, pos_tag: str) -> int:
        """Count tokens with specific POS tag."""
        return sum(1 for token in doc if token.pos_ == pos_tag)

    @staticmethod
    def count_dependency(doc, dep_label: str) -> int:
        """Count tokens with specific dependency label."""
        return sum(1 for token in doc if token.dep_ == dep_label)

    @staticmethod
    def sentence_length(doc) -> int:
        """Return sentence length (excluding spaces and punctuation)."""
        return len(
            [token for token in doc if not token.is_space and not token.is_punct]
        )

    @staticmethod
    def noun_phrase_count(doc) -> int:
        """Count noun phrases."""
        return len(list(doc.noun_chunks))

    @staticmethod
    def verb_phrase_complexity(doc) -> int:
        """Measure verb phrase complexity."""
        verb_count = sum(1 for token in doc if token.pos_ == "VERB")
        aux_count = sum(1 for token in doc if token.pos_ == "AUX")
        return verb_count + aux_count

    @staticmethod
    def prepositional_phrase_count(doc) -> int:
        """Count prepositional phrases."""
        return sum(1 for token in doc if token.dep_ == "prep")

    @staticmethod
    def subordinate_clause_count(doc) -> int:
        """Count subordinate clauses."""
        return sum(
            1
            for token in doc
            if token.dep_ in ["ccomp", "xcomp", "advcl", "acl", "relcl"]
        )

    @staticmethod
    def coordination_count(doc) -> int:
        """Count coordination structures."""
        return sum(1 for token in doc if token.dep_ == "conj")

    @staticmethod
    def has_garden_path_structure(doc) -> bool:
        """Detect potential garden path structures."""
        # Look for past participles that could be main verbs or reduced relatives
        for token in doc:
            if (
                token.pos_ == "VERB"
                and token.tag_ == "VBN"
                and token.head.pos_ == "NOUN"
                and token.i > token.head.i
            ):
                return True
        return False

    @staticmethod
    def ambiguous_word_count(doc) -> int:
        """Count potentially ambiguous words (words with multiple POS possibilities)."""
        ambiguous_words = {
            "saw",
            "bat",
            "duck",
            "bark",
            "bank",
            "bear",
            "fair",
            "light",
            "right",
        }
        return sum(1 for token in doc if token.lemma_.lower() in ambiguous_words)

    @staticmethod
    def attachment_ambiguity_score(doc) -> float:
        """Score potential PP attachment ambiguity."""
        score = 0.0
        prepositions = [token for token in doc if token.pos_ == "ADP"]

        for prep in prepositions:
            # Count potential attachment sites (nouns and verbs before the preposition)
            potential_heads = [
                token
                for token in doc
                if token.i < prep.i and token.pos_ in ["NOUN", "VERB"]
            ]
            if len(potential_heads) > 1:
                score += len(potential_heads) - 1

        return score / len(doc) if len(doc) > 0 else 0.0

    @staticmethod
    def syntactic_complexity(doc) -> float:
        """Measure overall syntactic complexity."""
        dep_types = len(set(token.dep_ for token in doc))
        pos_types = len(set(token.pos_ for token in doc))
        return (dep_types + pos_types) / len(doc) if len(doc) > 0 else 0.0

    @staticmethod
    def lexical_diversity(doc) -> float:
        """Measure lexical diversity (type-token ratio)."""
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and not token.is_punct
        ]
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)

    @staticmethod
    def has_modal_verbs(doc) -> bool:
        """Check for modal verbs that can create ambiguity."""
        modals = {"can", "could", "may", "might", "must", "should", "would", "will"}
        return any(token.lemma_.lower() in modals for token in doc)

    @staticmethod
    def passive_voice_count(doc) -> int:
        """Count passive voice constructions."""
        count = 0
        for token in doc:
            if (
                token.pos_ == "AUX"
                and token.lemma_ == "be"
                and any(
                    child.pos_ == "VERB" and child.tag_ == "VBN"
                    for child in token.children
                )
            ):
                count += 1
        return count


# --- 4. GENETIC PROGRAMMING SETUP ---

# Define the primitive set using typed GP
pset = gp.PrimitiveSetTyped("main", [object], bool)
pset.renameArguments(ARG0="doc")

# Boolean operations
pset.addPrimitive(operator.and_, [bool, bool], bool, name="AND")
pset.addPrimitive(operator.or_, [bool, bool], bool, name="OR")
pset.addPrimitive(operator.not_, [bool], bool, name="NOT")


# Comparison operations for numeric features
def safe_gt(a, b):
    """Safe greater than comparison."""
    try:
        return float(a) > float(b)
    except (ValueError, TypeError):
        return False


def safe_lt(a, b):
    """Safe less than comparison."""
    try:
        return float(a) < float(b)
    except (ValueError, TypeError):
        return False


def safe_eq(a, b):
    """Safe equality comparison."""
    try:
        return abs(float(a) - float(b)) < 0.001
    except (ValueError, TypeError):
        return a == b


pset.addPrimitive(safe_gt, [float, float], bool, name="GT")
pset.addPrimitive(safe_lt, [float, float], bool, name="LT")
pset.addPrimitive(safe_eq, [float, float], bool, name="EQ")


# Feature extraction primitives with proper typing
def add_numeric_feature(name: str, feature_func):
    """Helper to add numeric feature extraction primitives."""
    pset.addPrimitive(feature_func, [object], float, name=name)


def add_boolean_feature(name: str, feature_func):
    """Helper to add boolean feature extraction primitives."""
    pset.addPrimitive(feature_func, [object], bool, name=name)


# Add numeric linguistic features
add_numeric_feature(
    "noun_count", lambda doc: float(LinguisticFeatures.count_pos_tag(doc, "NOUN"))
)
add_numeric_feature(
    "verb_count", lambda doc: float(LinguisticFeatures.count_pos_tag(doc, "VERB"))
)
add_numeric_feature(
    "adj_count", lambda doc: float(LinguisticFeatures.count_pos_tag(doc, "ADJ"))
)
add_numeric_feature(
    "adv_count", lambda doc: float(LinguisticFeatures.count_pos_tag(doc, "ADV"))
)
add_numeric_feature(
    "prep_count", lambda doc: float(LinguisticFeatures.count_pos_tag(doc, "ADP"))
)

add_numeric_feature(
    "sent_length", lambda doc: float(LinguisticFeatures.sentence_length(doc))
)
add_numeric_feature(
    "np_count", lambda doc: float(LinguisticFeatures.noun_phrase_count(doc))
)
add_numeric_feature(
    "vp_complexity", lambda doc: float(LinguisticFeatures.verb_phrase_complexity(doc))
)
add_numeric_feature(
    "pp_count", lambda doc: float(LinguisticFeatures.prepositional_phrase_count(doc))
)
add_numeric_feature(
    "subord_count", lambda doc: float(LinguisticFeatures.subordinate_clause_count(doc))
)
add_numeric_feature(
    "coord_count", lambda doc: float(LinguisticFeatures.coordination_count(doc))
)
add_numeric_feature(
    "passive_count", lambda doc: float(LinguisticFeatures.passive_voice_count(doc))
)

add_numeric_feature(
    "ambig_words", lambda doc: float(LinguisticFeatures.ambiguous_word_count(doc))
)
add_numeric_feature(
    "attach_score", lambda doc: LinguisticFeatures.attachment_ambiguity_score(doc)
)
add_numeric_feature(
    "syntax_complex", lambda doc: LinguisticFeatures.syntactic_complexity(doc)
)
add_numeric_feature(
    "lex_diversity", lambda doc: LinguisticFeatures.lexical_diversity(doc)
)

# Add boolean linguistic features
add_boolean_feature("has_garden_path", LinguisticFeatures.has_garden_path_structure)
add_boolean_feature("has_modals", LinguisticFeatures.has_modal_verbs)

# Dependency-specific features
add_numeric_feature(
    "pobj_count", lambda doc: float(LinguisticFeatures.count_dependency(doc, "pobj"))
)
add_numeric_feature(
    "nsubj_count", lambda doc: float(LinguisticFeatures.count_dependency(doc, "nsubj"))
)
add_numeric_feature(
    "dobj_count", lambda doc: float(LinguisticFeatures.count_dependency(doc, "dobj"))
)
add_numeric_feature(
    "prep_dep_count",
    lambda doc: float(LinguisticFeatures.count_dependency(doc, "prep")),
)

# Add terminals (constants) with proper typing
pset.addTerminal(True, bool, name="True")
pset.addTerminal(False, bool, name="False")

# Add numeric constants
for i in range(0, 8):
    pset.addTerminal(float(i), float, name=f"const_{i}")

# Add float constants for thresholds
for f in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    pset.addTerminal(f, float, name=f"float_{str(f).replace('.', '_')}")


# Ephemeral constants with proper typing
def rand_float():
    return round(random.uniform(0.0, 5.0), 2)


def rand_bool():
    return random.choice([True, False])


pset.addEphemeralConstant("rand_float", rand_float, float)
pset.addEphemeralConstant("rand_bool", rand_bool, bool)

# --- 5. COMPLEXITY CONFIGURATION ---


class ComplexityConfig:
    """Configuration class for controlling rule complexity."""

    def __init__(self, complexity_preference="balanced"):
        """
        Initialize complexity configuration.

        Args:
            complexity_preference: "simple", "balanced", "complex", or "custom"
        """
        self.complexity_preference = complexity_preference
        self._set_parameters()

    def _set_parameters(self):
        """Set parameters based on complexity preference."""
        if self.complexity_preference == "simple":
            # Encourage simple rules
            self.min_complexity = 1
            self.max_complexity = 5
            self.optimal_complexity = 3
            self.complexity_weight = -0.1  # Penalty for complexity
            self.diversity_weight = 0.02
            self.min_tree_size = 1
            self.max_tree_size = 6
            self.optimal_depth = 2

        elif self.complexity_preference == "complex":
            # Encourage complex rules
            self.min_complexity = 5
            self.max_complexity = 20
            self.optimal_complexity = 12
            self.complexity_weight = 0.15  # Bonus for complexity
            self.diversity_weight = 0.1
            self.min_tree_size = 4
            self.max_tree_size = 12
            self.optimal_depth = 6

        elif self.complexity_preference == "balanced":
            # Default balanced approach
            self.min_complexity = 3
            self.max_complexity = 15
            self.optimal_complexity = 7
            self.complexity_weight = 0.1
            self.diversity_weight = 0.05
            self.min_tree_size = 3
            self.max_tree_size = 8
            self.optimal_depth = 4

        elif self.complexity_preference == "explore":
            # Maximum exploration mode
            self.min_complexity = 1
            self.max_complexity = 30
            self.optimal_complexity = 10  # No strong preference
            self.complexity_weight = 0.05  # Small complexity bonus
            self.diversity_weight = 0.2  # High diversity bonus
            self.min_tree_size = 1
            self.max_tree_size = 15
            self.optimal_depth = 5
            self.novelty_weight = 0.3  # New parameter for exploration
            self.mutation_strength = 0.7  # Higher mutation for exploration
            self.population_multiplier = 3  # Larger populations
            self.selection_pressure = 0.3  # Lower selection pressure

        else:  # custom - use defaults but allow override
            self.min_complexity = 3
            self.max_complexity = 15
            self.optimal_complexity = 7
            self.complexity_weight = 0.1
            self.diversity_weight = 0.05
            self.min_tree_size = 3
            self.max_tree_size = 8
            self.optimal_depth = 4

        # Additional exploration parameters (set defaults)
        if not hasattr(self, "novelty_weight"):
            self.novelty_weight = 0.1
        if not hasattr(self, "mutation_strength"):
            self.mutation_strength = 0.3
        if not hasattr(self, "population_multiplier"):
            self.population_multiplier = 1
        if not hasattr(self, "selection_pressure"):
            self.selection_pressure = 0.6

    def set_custom_params(self, **kwargs):
        """Set custom complexity parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown parameter {key}")


# Global complexity configuration
COMPLEXITY_CONFIG = ComplexityConfig("balanced")

# Global storage for rule diversity tracking
RULE_ARCHIVE = []
RULE_SIGNATURES = set()

# --- 6. ENHANCED FITNESS EVALUATION ---


def evaluate_individual(individual) -> Tuple[float]:
    """
    Enhanced fitness evaluation with complexity control and diversity tracking.
    Returns tuple with complexity-adjusted fitness score.
    """
    try:
        # Compile the individual into a callable function
        rule_func = toolbox.compile(expr=individual)

        predictions = []
        true_labels = []
        errors = 0

        for doc, is_ambiguous in PARSED_DATASET:
            try:
                prediction = rule_func(doc)
                # Ensure prediction is boolean
                if not isinstance(prediction, bool):
                    prediction = bool(prediction)

                predictions.append(prediction)
                true_labels.append(is_ambiguous)

            except Exception as e:
                # If rule fails, count as error and assign False
                predictions.append(False)
                true_labels.append(is_ambiguous)
                errors += 1

        # If too many errors, return very low fitness
        if errors > len(PARSED_DATASET) * 0.5:
            return (0.01,)

        # Calculate metrics using sklearn
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average="weighted"
        )

        # Debug: print some info for first few evaluations
        if hasattr(evaluate_individual, "call_count"):
            evaluate_individual.call_count += 1
        else:
            evaluate_individual.call_count = 1

        if evaluate_individual.call_count <= 3:
            print(f"Debug - Rule: {str(individual)[:50]}...")
            print(f"Debug - Predictions: {predictions[:10]}...")
            print(f"Debug - True labels: {true_labels[:10]}...")
            print(f"Debug - F1: {f1}, Errors: {errors}, Size: {len(individual)}")

        # Penalize trivial solutions (always same prediction)
        unique_predictions = len(set(predictions))
        if unique_predictions == 1:
            return (0.05,)

        # NOVELTY AND DIVERSITY TRACKING
        rule_signature = create_rule_signature(individual, predictions)
        novelty_bonus = calculate_novelty_bonus(rule_signature, individual)

        # COMPLEXITY CONTROL MECHANISMS

        # 1. Calculate rule complexity metrics
        rule_size = len(individual)  # Number of nodes
        rule_depth = individual.height  # Tree depth

        # Count unique features used (feature diversity)
        feature_nodes = [
            node.name
            for node in individual
            if hasattr(node, "name")
            and any(
                feature in node.name
                for feature in [
                    "noun_count",
                    "verb_count",
                    "adj_count",
                    "adv_count",
                    "prep_count",
                    "sent_length",
                    "np_count",
                    "vp_complexity",
                    "pp_count",
                    "subord_count",
                    "coord_count",
                    "passive_count",
                    "has_garden_path",
                    "has_modals",
                    "ambig_words",
                    "attach_score",
                    "syntax_complex",
                    "lex_diversity",
                    "pobj_count",
                    "nsubj_count",
                    "dobj_count",
                    "prep_dep_count",
                ]
            )
        ]
        unique_features = len(set(feature_nodes))

        # 2. Complexity bonuses and penalties (using configuration)
        MIN_COMPLEXITY = COMPLEXITY_CONFIG.min_complexity
        MAX_COMPLEXITY = COMPLEXITY_CONFIG.max_complexity
        OPTIMAL_COMPLEXITY = COMPLEXITY_CONFIG.optimal_complexity

        # Size-based adjustment
        if rule_size < MIN_COMPLEXITY:
            complexity_penalty = 0.8  # Penalize overly simple rules
        elif rule_size > MAX_COMPLEXITY:
            complexity_penalty = 0.9  # Penalize overly complex rules
        else:
            # Reward rules near optimal complexity
            distance_from_optimal = abs(rule_size - OPTIMAL_COMPLEXITY)
            complexity_penalty = 1.0 + (
                COMPLEXITY_CONFIG.complexity_weight
                * (1.0 - distance_from_optimal / OPTIMAL_COMPLEXITY)
            )

        # 3. Feature diversity bonus
        diversity_bonus = 1.0 + (
            COMPLEXITY_CONFIG.diversity_weight * unique_features
        )  # Bonus for using more features

        # 4. Depth-based adjustment (encourage moderate depth)
        OPTIMAL_DEPTH = COMPLEXITY_CONFIG.optimal_depth
        if rule_depth < 2:
            depth_penalty = 0.9  # Too shallow
        elif rule_depth > 8:
            depth_penalty = 0.85  # Too deep
        else:
            depth_penalty = 1.0 + (
                0.05 * (1.0 - abs(rule_depth - OPTIMAL_DEPTH) / OPTIMAL_DEPTH)
            )

        # 5. Performance-complexity trade-off
        base_fitness = f1

        # Require minimum performance before complexity bonuses apply
        if f1 < 0.3:
            # For poor performing rules, focus purely on performance
            final_fitness = base_fitness
        else:
            # For decent performing rules, apply complexity adjustments
            final_fitness = (
                base_fitness * complexity_penalty * diversity_bonus * depth_penalty
            )

        # 6. Bonus for balanced precision/recall
        if precision > 0 and recall > 0:
            balance_bonus = 1.0 + (0.1 * (1.0 - abs(precision - recall)))
            final_fitness *= balance_bonus

        # 7. Apply novelty bonus for exploration
        final_fitness *= 1.0 + COMPLEXITY_CONFIG.novelty_weight * novelty_bonus

        return (final_fitness,)

    except Exception as e:
        print(f"Compilation error: {e}")
        return (0.0,)


def create_rule_signature(individual, predictions):
    """Create a signature for rule diversity tracking."""
    # Combine structural and behavioral signatures
    structure_sig = str(individual)
    behavior_sig = tuple(predictions)

    # Extract key features used
    feature_nodes = [
        node.name
        for node in individual
        if hasattr(node, "name")
        and any(
            feature in node.name
            for feature in [
                "noun_count",
                "verb_count",
                "adj_count",
                "adv_count",
                "prep_count",
                "sent_length",
                "np_count",
                "vp_complexity",
                "pp_count",
                "subord_count",
                "coord_count",
                "passive_count",
                "has_garden_path",
                "has_modals",
                "ambig_words",
                "attach_score",
                "syntax_complex",
                "lex_diversity",
                "pobj_count",
                "nsubj_count",
                "dobj_count",
                "prep_dep_count",
            ]
        )
    ]
    feature_sig = tuple(sorted(set(feature_nodes)))

    return {
        "structure": structure_sig,
        "behavior": behavior_sig,
        "features": feature_sig,
        "size": len(individual),
        "depth": individual.height,
    }


def calculate_novelty_bonus(rule_signature, individual):
    """Calculate novelty bonus based on rule uniqueness."""
    global RULE_ARCHIVE, RULE_SIGNATURES

    # Behavioral novelty (unique prediction patterns)
    behavior_key = rule_signature["behavior"]
    behavior_novelty = (
        1.0 if behavior_key not in {r["behavior"] for r in RULE_ARCHIVE} else 0.3
    )

    # Structural novelty (unique rule structures)
    structure_key = rule_signature["structure"]
    structure_novelty = (
        1.0 if structure_key not in {r["structure"] for r in RULE_ARCHIVE} else 0.2
    )

    # Feature combination novelty
    feature_key = rule_signature["features"]
    feature_novelty = (
        1.0 if feature_key not in {r["features"] for r in RULE_ARCHIVE} else 0.4
    )

    # Size/depth diversity
    size_depth_key = (rule_signature["size"], rule_signature["depth"])
    size_novelty = (
        1.0
        if size_depth_key not in {(r["size"], r["depth"]) for r in RULE_ARCHIVE}
        else 0.6
    )

    # Store rule in archive (limit size to prevent memory issues)
    if len(RULE_ARCHIVE) < 1000:  # Limit archive size
        RULE_ARCHIVE.append(rule_signature)

    # Combined novelty score
    novelty_score = (
        behavior_novelty + structure_novelty + feature_novelty + size_novelty
    ) / 4.0
    return novelty_score


# Define fitness and individual classes
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# Setup enhanced toolbox with complexity control
toolbox = base.Toolbox()

# Control initial complexity based on configuration
toolbox.register(
    "expr",
    gp.genHalfAndHalf,
    pset=pset,
    min_=COMPLEXITY_CONFIG.min_tree_size,
    max_=COMPLEXITY_CONFIG.max_tree_size,
    type_=bool,
)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# Register genetic operators with complexity awareness
toolbox.register("evaluate", evaluate_individual)


# Multi-objective selection considering fitness, complexity, and diversity
def diversity_aware_selection(individuals, k):
    """Selection that maximizes diversity while maintaining quality."""
    if len(individuals) <= k:
        return individuals

    # For exploration mode, use more diverse selection
    if COMPLEXITY_CONFIG.complexity_preference == "explore":
        return exploration_selection(individuals, k)
    else:
        return complexity_aware_selection(individuals, k)


def exploration_selection(individuals, k):
    """Maximum diversity selection for exploration mode."""
    selected = []
    remaining = list(individuals)

    # Always include the best individual
    best = max(remaining, key=lambda x: x.fitness.values[0])
    selected.append(best)
    remaining.remove(best)
    k -= 1

    # Select remaining based on diversity metrics
    while len(selected) < k and remaining:
        best_diversity_score = -1
        best_candidate = None

        for candidate in remaining:
            diversity_score = calculate_diversity_score(candidate, selected)
            if diversity_score > best_diversity_score:
                best_diversity_score = diversity_score
                best_candidate = candidate

        if best_candidate:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
        else:
            break

    # Fill remaining slots with random selection for maximum exploration
    while len(selected) < k and remaining:
        selected.append(random.choice(remaining))
        remaining.remove(selected[-1])

    return selected


def calculate_diversity_score(candidate, selected_individuals):
    """Calculate how diverse a candidate is compared to already selected individuals."""
    if not selected_individuals:
        return 1.0

    # Structural diversity
    candidate_str = str(candidate)
    structure_distances = [
        string_distance(candidate_str, str(selected))
        for selected in selected_individuals
    ]
    structure_diversity = sum(structure_distances) / len(structure_distances)

    # Size diversity
    candidate_size = len(candidate)
    size_distances = [
        abs(candidate_size - len(selected)) for selected in selected_individuals
    ]
    size_diversity = sum(size_distances) / len(size_distances) if size_distances else 0

    # Depth diversity
    candidate_depth = candidate.height
    depth_distances = [
        abs(candidate_depth - selected.height) for selected in selected_individuals
    ]
    depth_diversity = (
        sum(depth_distances) / len(depth_distances) if depth_distances else 0
    )

    # Combined diversity score
    return (structure_diversity + size_diversity + depth_diversity) / 3.0


def string_distance(s1, s2):
    """Simple string distance metric."""
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)

    # Simple character-based distance
    max_len = max(len(s1), len(s2))
    common_chars = sum(1 for a, b in zip(s1, s2) if a == b)
    return (max_len - common_chars) / max_len


def complexity_aware_selection(individuals, k):
    """Selection that considers both fitness and complexity."""
    # Sort by fitness first
    sorted_by_fitness = sorted(
        individuals, key=lambda x: x.fitness.values[0], reverse=True
    )

    # Among top performers, prefer more complex solutions
    top_quarter = int(len(sorted_by_fitness) * 0.25)
    top_performers = sorted_by_fitness[: max(top_quarter, k)]

    if len(top_performers) <= k:
        return top_performers

    # Among top performers, select based on complexity diversity
    selected = []
    complexity_buckets = {}

    for ind in top_performers:
        complexity = len(ind)
        if complexity not in complexity_buckets:
            complexity_buckets[complexity] = []
        complexity_buckets[complexity].append(ind)

    # Select from different complexity levels
    complexities = sorted(complexity_buckets.keys())
    while len(selected) < k and complexities:
        for comp in complexities:
            if complexity_buckets[comp] and len(selected) < k:
                selected.append(complexity_buckets[comp].pop(0))
        complexities = [c for c in complexities if complexity_buckets[c]]

    return selected[:k]


toolbox.register("select", diversity_aware_selection)

# Enhanced crossover that can increase complexity
toolbox.register("mate", gp.cxOnePoint)


# Multiple mutation strategies with exploration support
def adaptive_mutation(individual, expr, pset):
    """Enhanced mutation that adapts based on complexity preference."""

    if COMPLEXITY_CONFIG.complexity_preference == "explore":
        # Maximum diversity mutations for exploration
        mutation_types = [
            "uniform",
            "insert",
            "shrink",
            "node_replacement",
            "subtree_crossover",
        ]
        weights = [0.3, 0.25, 0.15, 0.2, 0.1]  # Favor structure-changing mutations
    else:
        # Standard mutations
        mutation_types = ["uniform", "insert", "shrink"]
        weights = [0.5, 0.3, 0.2]

    mutation_type = random.choices(mutation_types, weights=weights)[0]

    if mutation_type == "uniform":
        return gp.mutUniform(individual, expr, pset)
    elif mutation_type == "insert":
        return gp.mutInsert(individual, pset)
    elif mutation_type == "shrink":
        return gp.mutShrink(individual)
    elif mutation_type == "node_replacement":
        return gp.mutNodeReplacement(individual, pset)
    elif mutation_type == "subtree_crossover":
        # Create a random individual and crossover with it
        random_individual = toolbox.individual()
        offspring1, offspring2 = gp.cxOnePoint(individual, random_individual)
        return (offspring1,)

    return gp.mutUniform(individual, expr, pset)


toolbox.register(
    "expr_mut",
    gp.genGrow,
    min_=max(1, COMPLEXITY_CONFIG.min_tree_size - 1),
    max_=COMPLEXITY_CONFIG.max_tree_size,
    type_=bool,
)
toolbox.register("mutate", adaptive_mutation, expr=toolbox.expr_mut, pset=pset)

# Adaptive bloat control - allow larger trees but with penalties
toolbox.decorate(
    "mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=12)
)
toolbox.decorate(
    "mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=12)
)

# --- 7. DETAILED EVALUATION ---


def detailed_evaluation(individual, dataset=None):
    """Perform detailed evaluation of an individual."""
    if dataset is None:
        dataset = PARSED_DATASET

    rule_func = toolbox.compile(expr=individual)
    predictions = []
    true_labels = []

    for doc, is_ambiguous in dataset:
        try:
            prediction = rule_func(doc)
            if not isinstance(prediction, bool):
                prediction = bool(prediction)
            predictions.append(prediction)
            true_labels.append(is_ambiguous)
        except Exception:
            predictions.append(False)
            true_labels.append(is_ambiguous)

    # Calculate all metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average="weighted"
    )
    accuracy = accuracy_score(true_labels, predictions)

    return {
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "predictions": predictions,
        "true_labels": true_labels,
    }


def main(complexity_preference="balanced", **complexity_params):
    """
    Enhanced main execution with complexity control.

    Args:
        complexity_preference: "simple", "balanced", "complex", or "custom"
        **complexity_params: Custom complexity parameters for "custom" mode
    """
    # Configure complexity settings
    global COMPLEXITY_CONFIG
    COMPLEXITY_CONFIG = ComplexityConfig(complexity_preference)
    if complexity_preference == "custom":
        COMPLEXITY_CONFIG.set_custom_params(**complexity_params)

    # Recreate toolbox with new complexity settings
    toolbox.unregister("expr")
    toolbox.unregister("expr_mut")
    toolbox.register(
        "expr",
        gp.genHalfAndHalf,
        pset=pset,
        min_=COMPLEXITY_CONFIG.min_tree_size,
        max_=COMPLEXITY_CONFIG.max_tree_size,
        type_=bool,
    )
    toolbox.register(
        "expr_mut",
        gp.genGrow,
        min_=max(1, COMPLEXITY_CONFIG.min_tree_size - 1),
        max_=COMPLEXITY_CONFIG.max_tree_size,
        type_=bool,
    )

    random.seed(42)
    np.random.seed(42)

    # Create initial population - adjust size based on complexity preference
    base_pop_size = 150
    pop_size = int(base_pop_size * COMPLEXITY_CONFIG.population_multiplier)

    # For exploration mode, use multiple initialization strategies
    if COMPLEXITY_CONFIG.complexity_preference == "explore":
        pop = create_diverse_population(pop_size)
        hof = tools.HallOfFame(50)  # Keep many more individuals for diversity
        print(
            f"Exploration mode: Using diverse initialization with {pop_size} individuals"
        )
    else:
        pop = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(10)  # Keep top 10 individuals

    # Enhanced statistics
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    print("=" * 60)
    print("ENHANCED GENETIC ALGORITHM FOR AMBIGUITY DETECTION")
    print("=" * 60)
    print(f"Complexity Preference: {COMPLEXITY_CONFIG.complexity_preference}")
    print(
        f"Target Complexity Range: {COMPLEXITY_CONFIG.min_complexity}-{COMPLEXITY_CONFIG.max_complexity} nodes"
    )
    print(f"Optimal Complexity: {COMPLEXITY_CONFIG.optimal_complexity} nodes")
    print(
        f"Tree Size Range: {COMPLEXITY_CONFIG.min_tree_size}-{COMPLEXITY_CONFIG.max_tree_size}"
    )
    print(f"Dataset size: {len(PARSED_DATASET)}")
    print(f"Ambiguous sentences: {sum(1 for _, label in SENTENCE_DATASET if label)}")
    print(
        f"Unambiguous sentences: {sum(1 for _, label in SENTENCE_DATASET if not label)}"
    )
    print(f"Population size: {pop_size}")
    print(f"Available primitives: {len(pset.primitives)}")
    print(f"Available terminals: {len(pset.terminals)}")
    print("\nStarting evolution...")

    # Run evolution with adaptive parameters based on complexity preference
    if COMPLEXITY_CONFIG.complexity_preference == "explore":
        # Exploration-focused parameters
        cxpb = 0.5  # Lower crossover, higher mutation for exploration
        mutpb = COMPLEXITY_CONFIG.mutation_strength  # High mutation rate
        ngen = 60  # More generations to explore thoroughly
        print(f"Exploration mode: cxpb={cxpb}, mutpb={mutpb}, ngen={ngen}")
    else:
        # Standard parameters
        cxpb = 0.7
        mutpb = 0.3
        ngen = 40

    pop, log = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=cxpb,
        mutpb=mutpb,
        ngen=ngen,
        stats=mstats,
        halloffame=hof,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("EVOLUTION COMPLETE - RESULTS ANALYSIS")
    print("=" * 60)

    # Analyze best individuals with complexity metrics
    for i, individual in enumerate(hof):
        print(f"\n--- RANK {i + 1} INDIVIDUAL ---")
        metrics = detailed_evaluation(individual)

        # Calculate complexity metrics
        rule_size = len(individual)
        rule_depth = individual.height
        feature_nodes = [
            node.name
            for node in individual
            if hasattr(node, "name")
            and any(
                feature in node.name
                for feature in [
                    "noun_count",
                    "verb_count",
                    "adj_count",
                    "adv_count",
                    "prep_count",
                    "sent_length",
                    "np_count",
                    "vp_complexity",
                    "pp_count",
                    "subord_count",
                    "coord_count",
                    "passive_count",
                    "has_garden_path",
                    "has_modals",
                    "ambig_words",
                    "attach_score",
                    "syntax_complex",
                    "lex_diversity",
                    "pobj_count",
                    "nsubj_count",
                    "dobj_count",
                    "prep_dep_count",
                ]
            )
        ]
        unique_features = len(set(feature_nodes))

        print(f"Composite Fitness: {individual.fitness.values[0]:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Rule Size: {rule_size} nodes")
        print(f"Rule Depth: {rule_depth}")
        print(f"Unique Features: {unique_features}")
        print(f"Features Used: {set(feature_nodes)}")
        print(f"Rule: {str(individual)}")

    # Test best rule on each sentence
    print("\n--- DETAILED TESTING OF BEST RULE ---")
    best_individual = hof[0]
    best_metrics = detailed_evaluation(best_individual)

    print(f"Best Rule: {str(best_individual)}")
    print("Performance Metrics:")
    print(f"  F1 Score: {best_metrics['f1_score']:.4f}")
    print(f"  Precision: {best_metrics['precision']:.4f}")
    print(f"  Recall: {best_metrics['recall']:.4f}")
    print(f"  Accuracy: {best_metrics['accuracy']:.4f}")

    print("\nPer-sentence Analysis:")
    for i, (sentence, true_label) in enumerate(SENTENCE_DATASET):
        pred = best_metrics["predictions"][i]
        status = "✓" if pred == true_label else "✗"
        ambig_type = "AMBIGUOUS" if true_label else "CLEAR"
        print(f"{status} [{ambig_type:9}] '{sentence}' -> {pred}")

    # Print exploration statistics
    if COMPLEXITY_CONFIG.complexity_preference == "explore":
        print("\n--- EXPLORATION STATISTICS ---")
        print(f"Total unique rules explored: {len(RULE_ARCHIVE)}")
        print(
            f"Unique behavioral patterns: {len(set(r['behavior'] for r in RULE_ARCHIVE))}"
        )
        print(
            f"Unique feature combinations: {len(set(r['features'] for r in RULE_ARCHIVE))}"
        )
        print(
            f"Size range: {min(r['size'] for r in RULE_ARCHIVE)} - {max(r['size'] for r in RULE_ARCHIVE)}"
        )
        print(
            f"Depth range: {min(r['depth'] for r in RULE_ARCHIVE)} - {max(r['depth'] for r in RULE_ARCHIVE)}"
        )

    return pop, log, hof


def create_diverse_population(pop_size):
    """Create a diverse initial population using multiple strategies."""
    population = []

    # Strategy 1: Standard generation (1/3 of population)
    for _ in range(pop_size // 3):
        population.append(toolbox.individual())

    # Strategy 2: Feature-focused generation (1/3 of population)
    for _ in range(pop_size // 3):
        individual = create_feature_focused_individual()
        population.append(individual)

    # Strategy 3: Size-diverse generation (remaining)
    remaining = pop_size - len(population)
    for i in range(remaining):
        # Create individuals with different target sizes
        target_size = 2 + (i * 3) % 15  # Sizes from 2 to 17
        individual = create_size_targeted_individual(target_size)
        population.append(individual)

    return population


def create_feature_focused_individual():
    """Create an individual that focuses on using specific features."""
    # Randomly select 1-3 features to focus on
    all_features = [
        "noun_count",
        "verb_count",
        "adj_count",
        "adv_count",
        "prep_count",
        "sent_length",
        "np_count",
        "vp_complexity",
        "pp_count",
        "subord_count",
        "coord_count",
        "passive_count",
        "has_garden_path",
        "has_modals",
        "ambig_words",
        "attach_score",
        "syntax_complex",
        "lex_diversity",
        "pobj_count",
        "nsubj_count",
        "dobj_count",
        "prep_dep_count",
    ]

    selected_features = random.sample(all_features, random.randint(1, 3))

    # Build a rule using these features
    expr = build_feature_expression(selected_features)
    return creator.Individual(expr)


def create_size_targeted_individual(target_size):
    """Create an individual targeting a specific size."""
    attempts = 0
    while attempts < 10:  # Limit attempts
        try:
            individual = toolbox.individual()
            if abs(len(individual) - target_size) <= 2:  # Close enough
                return individual
        except:
            pass
        attempts += 1

    # Fallback to standard individual
    return toolbox.individual()


def build_feature_expression(features):
    """Build an expression using specific features."""
    # This is a simplified version - in practice, you'd want more sophisticated construction
    try:
        # Create a simple comparison using the features
        if len(features) == 1:
            feature = features[0]
            return gp.PrimitiveTree.from_string(f"GT({feature}(doc), const_1)", pset)
        else:
            # Combine multiple features
            feature1, feature2 = features[0], features[1]
            return gp.PrimitiveTree.from_string(
                f"GT({feature1}(doc), {feature2}(doc))", pset
            )
    except:
        # Fallback to standard generation
        return toolbox.expr()


if __name__ == "__main__":
    # Example usage with different complexity preferences:

    # For simple rules (1-5 nodes):
    # main("simple")

    # For maximum exploration of diverse rules:
    main("explore")

    # For custom complexity settings:
    # main("custom", min_complexity=8, max_complexity=25, optimal_complexity=15)

    # For maximum exploration with custom parameters:
    # main("explore")  # Uses default exploration settings
    # main("custom", complexity_preference="explore", novelty_weight=0.5, mutation_strength=0.8, population_multiplier=4)
