"""
Benchmark Dataset for RAG vs GraphRAG Evaluation.

Contains 50+ curated questions with ground truth answers for:
- Entity Lookup (direct fact retrieval)
- Relationship Queries (graph traversal)
- Comparison Queries (multi-entity analysis)
- Aggregation Queries (category-level analysis)
- Reasoning Queries (combined criteria)
"""

from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum


class QuestionCategory(Enum):
    """Categories of benchmark questions."""
    ENTITY_LOOKUP = "entity_lookup"
    RELATIONSHIP = "relationship"
    COMPARISON = "comparison"
    AGGREGATION = "aggregation"
    REASONING = "reasoning"


class Difficulty(Enum):
    """Difficulty levels for questions."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class BenchmarkQuestion:
    """A single benchmark question with ground truth."""
    id: str
    question: str
    category: QuestionCategory
    difficulty: Difficulty
    ground_truth: str
    expected_entities: List[str]  # Entities that should be found
    expected_keywords: List[str]  # Keywords expected in answer
    requires_graph: bool  # True if graph context is essential
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "category": self.category.value,
            "difficulty": self.difficulty.value,
            "ground_truth": self.ground_truth,
            "expected_entities": self.expected_entities,
            "expected_keywords": self.expected_keywords,
            "requires_graph": self.requires_graph
        }


# ============================================================================
# BENCHMARK QUESTIONS (50+)
# ============================================================================

BENCHMARK_QUESTIONS: List[BenchmarkQuestion] = [
    # =========================================================================
    # CATEGORY 1: ENTITY LOOKUP (Direct Fact Retrieval)
    # =========================================================================
    BenchmarkQuestion(
        id="EL001",
        question="What is the price of Nike Air Max 270?",
        category=QuestionCategory.ENTITY_LOOKUP,
        difficulty=Difficulty.EASY,
        ground_truth="Nike Air Max 270 is a premium running shoe priced around ₹8000-₹15000.",
        expected_entities=["Nike", "Air Max 270"],
        expected_keywords=["price", "₹", "running", "shoe"],
        requires_graph=False
    ),
    BenchmarkQuestion(
        id="EL002",
        question="What brand manufactures the UltraBoost Light sneakers?",
        category=QuestionCategory.ENTITY_LOOKUP,
        difficulty=Difficulty.EASY,
        ground_truth="UltraBoost Light sneakers are manufactured by Adidas.",
        expected_entities=["Adidas", "UltraBoost Light"],
        expected_keywords=["Adidas", "brand", "manufactures"],
        requires_graph=True
    ),
    BenchmarkQuestion(
        id="EL003",
        question="What is the rating of Samsung Galaxy S24 Ultra?",
        category=QuestionCategory.ENTITY_LOOKUP,
        difficulty=Difficulty.EASY,
        ground_truth="Samsung Galaxy S24 Ultra typically has high ratings between 4.5 to 5.0 stars.",
        expected_entities=["Samsung", "Galaxy S24 Ultra"],
        expected_keywords=["rating", "stars", "4", "5"],
        requires_graph=False
    ),
    BenchmarkQuestion(
        id="EL004",
        question="What category does iPhone 15 Pro Max belong to?",
        category=QuestionCategory.ENTITY_LOOKUP,
        difficulty=Difficulty.EASY,
        ground_truth="iPhone 15 Pro Max belongs to the Electronics/Smartphones category.",
        expected_entities=["Apple", "iPhone 15 Pro Max"],
        expected_keywords=["category", "electronics", "smartphone"],
        requires_graph=True
    ),
    BenchmarkQuestion(
        id="EL005",
        question="What are the key features of Sony WH-1000XM5 headphones?",
        category=QuestionCategory.ENTITY_LOOKUP,
        difficulty=Difficulty.MEDIUM,
        ground_truth="Sony WH-1000XM5 features include active noise cancellation, long battery life, and premium sound quality.",
        expected_entities=["Sony", "WH-1000XM5"],
        expected_keywords=["noise cancellation", "battery", "sound"],
        requires_graph=False
    ),
    BenchmarkQuestion(
        id="EL006",
        question="How much does the Puma RS-X sneaker cost?",
        category=QuestionCategory.ENTITY_LOOKUP,
        difficulty=Difficulty.EASY,
        ground_truth="Puma RS-X sneakers are priced around ₹6000-₹9000.",
        expected_entities=["Puma", "RS-X"],
        expected_keywords=["price", "₹", "cost"],
        requires_graph=False
    ),
    BenchmarkQuestion(
        id="EL007",
        question="What is the description of Levi's Premium Jeans?",
        category=QuestionCategory.ENTITY_LOOKUP,
        difficulty=Difficulty.MEDIUM,
        ground_truth="Levi's Premium Jeans are classic fit jeans made from high-quality fabric.",
        expected_entities=["Levi's", "Jeans"],
        expected_keywords=["jeans", "fabric", "classic"],
        requires_graph=False
    ),
    BenchmarkQuestion(
        id="EL008",
        question="What brand makes the MacBook Air M2?",
        category=QuestionCategory.ENTITY_LOOKUP,
        difficulty=Difficulty.EASY,
        ground_truth="MacBook Air M2 is manufactured by Apple.",
        expected_entities=["Apple", "MacBook Air M2"],
        expected_keywords=["Apple", "laptop", "brand"],
        requires_graph=True
    ),
    BenchmarkQuestion(
        id="EL009",
        question="What is the price range for Reebok fitness shoes?",
        category=QuestionCategory.ENTITY_LOOKUP,
        difficulty=Difficulty.EASY,
        ground_truth="Reebok fitness shoes are typically priced between ₹2000-₹15000.",
        expected_entities=["Reebok"],
        expected_keywords=["price", "₹", "fitness", "shoes"],
        requires_graph=False
    ),
    BenchmarkQuestion(
        id="EL010",
        question="How many reviews does the Dell XPS 15 have?",
        category=QuestionCategory.ENTITY_LOOKUP,
        difficulty=Difficulty.MEDIUM,
        ground_truth="Dell XPS 15 laptops typically have hundreds to thousands of reviews.",
        expected_entities=["Dell", "XPS 15"],
        expected_keywords=["reviews", "laptop"],
        requires_graph=False
    ),
    BenchmarkQuestion(
        id="EL011",
        question="What are the specifictions of HP Spectre x360?",
        category=QuestionCategory.ENTITY_LOOKUP,
        difficulty=Difficulty.MEDIUM,
        ground_truth="HP Spectre x360 features high RAM (16/32GB), SSD storage, and varying processors.",
        expected_entities=["HP", "Spectre x360"],
        expected_keywords=["laptop", "RAM", "SSD"],
        requires_graph=False
    ),
    BenchmarkQuestion(
        id="EL012",
        question="What is the rating of Allen Solly shirts?",
        category=QuestionCategory.ENTITY_LOOKUP,
        difficulty=Difficulty.EASY,
        ground_truth="Allen Solly shirts typically have ratings between 3.5 and 4.6.",
        expected_entities=["Allen Solly"],
        expected_keywords=["rating", "stars", "shirt"],
        requires_graph=False
    ),

    # =========================================================================
    # CATEGORY 2: RELATIONSHIP QUERIES
    # =========================================================================
    BenchmarkQuestion(
        id="REL001",
        question="What products does Adidas manufacture?",
        category=QuestionCategory.RELATIONSHIP,
        difficulty=Difficulty.EASY,
        ground_truth="Adidas manufactures UltraBoost, Superstar, Stan Smith shoes, and various clothing items.",
        expected_entities=["Adidas"],
        expected_keywords=["shoes", "clothing", "UltraBoost"],
        requires_graph=True
    ),
    BenchmarkQuestion(
        id="REL002",
        question="Which brands are in the Footwear/Shoes category?",
        category=QuestionCategory.RELATIONSHIP,
        difficulty=Difficulty.MEDIUM,
        ground_truth="Footwear category includes Nike, Adidas, Puma, Reebok, New Balance, Skechers, Asics, Woodland.",
        expected_entities=["Nike", "Adidas", "Puma", "Skechers"],
        expected_keywords=["Nike", "Adidas", "Puma", "Skechers"],
        requires_graph=True
    ),
    BenchmarkQuestion(
        id="REL003",
        question="What categories does Nike have products in?",
        category=QuestionCategory.RELATIONSHIP,
        difficulty=Difficulty.MEDIUM,
        ground_truth="Nike has products in Shoes and Clothing categories.",
        expected_entities=["Nike"],
        expected_keywords=["shoes", "clothing", "categories"],
        requires_graph=True
    ),
    BenchmarkQuestion(
        id="REL004",
        question="List all products by Samsung",
        category=QuestionCategory.RELATIONSHIP,
        difficulty=Difficulty.EASY,
        ground_truth="Samsung products include Galaxy smartphones and Galaxy Book laptops.",
        expected_entities=["Samsung"],
        expected_keywords=["Galaxy", "smartphone", "laptop"],
        requires_graph=True
    ),
    BenchmarkQuestion(
        id="REL005",
        question="Which products belong to the Electronics category?",
        category=QuestionCategory.RELATIONSHIP,
        difficulty=Difficulty.MEDIUM,
        ground_truth="Electronics category includes Smartphones, Laptops, and Headphones.",
        expected_entities=[],
        expected_keywords=["smartphone", "laptop", "headphone", "electronics"],
        requires_graph=True
    ),
    BenchmarkQuestion(
        id="REL006",
        question="What is the relationship between Apple and iPhone 15?",
        category=QuestionCategory.RELATIONSHIP,
        difficulty=Difficulty.EASY,
        ground_truth="Apple is the brand that manufactures iPhone 15 smartphones.",
        expected_entities=["Apple", "iPhone 15"],
        expected_keywords=["manufactures", "brand", "Apple", "iPhone"],
        requires_graph=True
    ),
    BenchmarkQuestion(
        id="REL007",
        question="Which brands compete with Nike in running shoes?",
        category=QuestionCategory.RELATIONSHIP,
        difficulty=Difficulty.HARD,
        ground_truth="Nike competes with Adidas, Puma, New Balance, Asics, and Reebok in the shoe category.",
        expected_entities=["Nike", "Adidas", "Puma"],
        expected_keywords=["Adidas", "Puma", "compete"],
        requires_graph=True
    ),
    BenchmarkQuestion(
        id="REL008",
        question="What products are in the price range ₹20000-₹50000?",
        category=QuestionCategory.RELATIONSHIP,
        difficulty=Difficulty.MEDIUM,
        ground_truth="Products in this range include mid-range laptops, premium headphones, and some smartphones.",
        expected_entities=[],
        expected_keywords=["laptop", "headphone", "smartphone"],
        requires_graph=True
    ),
    BenchmarkQuestion(
        id="REL009",
        question="List brands that sell products under ₹5000",
        category=QuestionCategory.RELATIONSHIP,
        difficulty=Difficulty.MEDIUM,
        ground_truth="Brands with budget products include Boat, various Clothing brands, Grocery, and Stationery brands.",
        expected_entities=["Boat"],
        expected_keywords=["budget", "clothing", "stationery"],
        requires_graph=True
    ),
    BenchmarkQuestion(
        id="REL010",
        question="What products does Apple make besides iPhones?",
        category=QuestionCategory.RELATIONSHIP,
        difficulty=Difficulty.MEDIUM,
        ground_truth="Apple makes MacBook laptops (Air/Pro) and AirPods headphones besides iPhones.",
        expected_entities=["Apple", "MacBook", "AirPods"],
        expected_keywords=["MacBook", "AirPods", "laptop", "headphone"],
        requires_graph=True
    ),
    BenchmarkQuestion(
        id="REL011",
        question="Which category has the most expensive products?",
        category=QuestionCategory.RELATIONSHIP,
        difficulty=Difficulty.HARD,
        ground_truth="Laptops (Electronics) typically have the highest price points, reaching up to ₹250,000.",
        expected_entities=["Laptops"],
        expected_keywords=["laptop", "expensive", "250000"],
        requires_graph=True
    ),
    BenchmarkQuestion(
        id="REL012",
        question="What brands are connected to the Clothing category?",
        category=QuestionCategory.RELATIONSHIP,
        difficulty=Difficulty.MEDIUM,
        ground_truth="Clothing brands include Levi's, Allen Solly, Van Heusen, Zara, H&M, Nike, Adidas, etc.",
        expected_entities=["Levi's", "Zara", "Nike"],
        expected_keywords=["Levi's", "clothing", "brands"],
        requires_graph=True
    ),

    # =========================================================================
    # CATEGORY 3: COMPARISON QUERIES
    # =========================================================================
    BenchmarkQuestion(
        id="CMP001",
        question="Compare Nike and Adidas shoes",
        category=QuestionCategory.COMPARISON,
        difficulty=Difficulty.MEDIUM,
        ground_truth="Nike offers Air Max and Pegasus, while Adidas offers UltraBoost and Superstar. Both have similar pricing and ratings.",
        expected_entities=["Nike", "Adidas"],
        expected_keywords=["Nike", "Adidas", "running", "compare"],
        requires_graph=True
    ),
    BenchmarkQuestion(
        id="CMP002",
        question="Which is better rated: Samsung or Apple smartphones?",
        category=QuestionCategory.COMPARISON,
        difficulty=Difficulty.MEDIUM,
        ground_truth="Both Samsung (Galaxy) and Apple (iPhone) have high ratings (4.0-5.0), with premium models performing similarly.",
        expected_entities=["Samsung", "Apple"],
        expected_keywords=["Samsung", "Apple", "rating", "better"],
        requires_graph=True
    ),
    BenchmarkQuestion(
        id="CMP003",
        question="Compare prices of Sony and Bose headphones",
        category=QuestionCategory.COMPARISON,
        difficulty=Difficulty.MEDIUM,
        ground_truth="Sony (WH-1000XM5) and Bose (QuietComfort) are similarly priced premium headphones in the ₹20k-₹35k range.",
        expected_entities=["Sony", "Bose"],
        expected_keywords=["Sony", "Bose", "price", "headphones"],
        requires_graph=True
    ),
    BenchmarkQuestion(
        id="CMP004",
        question="Nike vs Puma: which has more varied shoe models?",
        category=QuestionCategory.COMPARISON,
        difficulty=Difficulty.HARD,
        ground_truth="Nike has Air Max, Air Force, Pegasus, Jordan. Puma has RS-X, Nitro, Suede. Both offer good variety.",
        expected_entities=["Nike", "Puma"],
        expected_keywords=["Nike", "Puma", "variety"],
        requires_graph=True
    ),
    BenchmarkQuestion(
        id="CMP005",
        question="Compare Dell and HP laptops",
        category=QuestionCategory.COMPARISON,
        difficulty=Difficulty.MEDIUM,
        ground_truth="Dell (XPS, Inspiron) and HP (Spectre, Pavilion) both offer laptops ranging from budget to premium.",
        expected_entities=["Dell", "HP"],
        expected_keywords=["Dell", "HP", "laptop", "compare"],
        requires_graph=True
    ),
    BenchmarkQuestion(
        id="CMP006",
        question="Which brand has higher ratings: Reebok or New Balance?",
        category=QuestionCategory.COMPARISON,
        difficulty=Difficulty.MEDIUM,
        ground_truth="Both Reebok and New Balance have strong customer reviews, often averaging above 4.0 stars.",
        expected_entities=["Reebok", "New Balance"],
        expected_keywords=["Reebok", "New Balance", "reviews"],
        requires_graph=True
    ),
    BenchmarkQuestion(
        id="CMP007",
        question="Compare Levi's and Allen Solly clothing",
        category=QuestionCategory.COMPARISON,
        difficulty=Difficulty.MEDIUM,
        ground_truth="Levi's is known for Jeans/Denim, while Allen Solly focuses more on Shirts and Trousers/Workwear.",
        expected_entities=["Levi's", "Allen Solly"],
        expected_keywords=["Levi's", "Allen Solly", "jeans", "shirt"],
        requires_graph=True
    ),
    BenchmarkQuestion(
        id="CMP008",
        question="Dell vs Apple: which laptop is more expensive?",
        category=QuestionCategory.COMPARISON,
        difficulty=Difficulty.HARD,
        ground_truth="Apple MacBooks (Air/Pro) are consistently premium priced. Dell has both budget (Inspiron) and premium (XPS). Apple average is higher.",
        expected_entities=["Dell", "Apple"],
        expected_keywords=["Dell", "Apple", "expensive", "laptop"],
        requires_graph=True
    ),
    BenchmarkQuestion(
        id="CMP009",
        question="Compare features of Adidas UltraBoost and Nike Air Max 270",
        category=QuestionCategory.COMPARISON,
        difficulty=Difficulty.HARD,
        ground_truth="UltraBoost emphasizes Boost cushioning. Air Max 270 emphasizes Air unit cushioning. Both are top-tier runners.",
        expected_entities=["Adidas", "Nike", "UltraBoost", "Air Max"],
        expected_keywords=["cushioning", "features", "running"],
        requires_graph=True
    ),
    BenchmarkQuestion(
        id="CMP010",
        question="Which is more affordable: Boat or Sony headphones?",
        category=QuestionCategory.COMPARISON,
        difficulty=Difficulty.EASY,
        ground_truth="Boat headphones are significantly more affordable (budget segment) compared to Sony's premium lineup.",
        expected_entities=["Boat", "Sony"],
        expected_keywords=["Boat", "Sony", "affordable", "price"],
        requires_graph=True
    ),

    # =========================================================================
    # CATEGORY 4: AGGREGATION QUERIES
    # =========================================================================
    BenchmarkQuestion(
        id="AGG001",
        question="What is the average price of Shoes?",
        category=QuestionCategory.AGGREGATION,
        difficulty=Difficulty.MEDIUM,
        ground_truth="Shoes average around ₹4000-₹8000, dependent on brand and type.",
        expected_entities=[],
        expected_keywords=["average", "price", "shoes", "₹"],
        requires_graph=False
    ),
    BenchmarkQuestion(
        id="AGG002",
        question="How many brands are in the Electronics category?",
        category=QuestionCategory.AGGREGATION,
        difficulty=Difficulty.MEDIUM,
        ground_truth="Electronics includes many brands: Apple, Samsung, OnePlus, Dell, HP, Sony, Bose, etc.",
        expected_entities=["Samsung", "Apple", "Sony"],
        expected_keywords=["brands", "electronics"],
        requires_graph=True
    ),
    BenchmarkQuestion(
        id="AGG003",
        question="What is the price range for premium headphones?",
        category=QuestionCategory.AGGREGATION,
        difficulty=Difficulty.EASY,
        ground_truth="Premium headphones (Sony, Bose, Apple) typically range from ₹15000 to ₹40000.",
        expected_entities=["Sony", "Bose"],
        expected_keywords=["premium", "headphones", "price"],
        requires_graph=False
    ),
    BenchmarkQuestion(
        id="AGG004",
        question="Which category has the highest rated products?",
        category=QuestionCategory.AGGREGATION,
        difficulty=Difficulty.MEDIUM,
        ground_truth="Premium Electronics (like Laptops/Phones) often have high ratings, as do premium Shoes.",
        expected_entities=[],
        expected_keywords=["highest", "rating", "category"],
        requires_graph=True
    ),
    BenchmarkQuestion(
        id="AGG005",
        question="What is the total number of products in Clothing?",
        category=QuestionCategory.AGGREGATION,
        difficulty=Difficulty.MEDIUM,
        ground_truth="The Clothing category contains products from many brands like Levis, Zara, etc.",
        expected_entities=["Clothing"],
        expected_keywords=["clothing", "products"],
        requires_graph=True
    ),

    # =========================================================================
    # CATEGORY 5: REASONING QUERIES
    # =========================================================================
    BenchmarkQuestion(
        id="RSN001",
        question="Recommend a laptop under ₹80000",
        category=QuestionCategory.REASONING,
        difficulty=Difficulty.MEDIUM,
        ground_truth="Consider Dell Inspiron, HP Pavilion, or basic MacBook Air M1 models (if on sale/refurb).",
        expected_entities=["Dell", "HP", "Apple"],
        expected_keywords=["laptop", "recommend", "₹80000"],
        requires_graph=True
    ),
    BenchmarkQuestion(
        id="RSN002",
        question="Best budget smartphone with good ratings?",
        category=QuestionCategory.REASONING,
        difficulty=Difficulty.MEDIUM,
        ground_truth="Samsung Galaxy M34, OnePlus Nord CE 3 Lite, or Realme phones are good budget options.",
        expected_entities=["Samsung", "OnePlus", "Realme"],
        expected_keywords=["budget", "smartphone", "rating"],
        requires_graph=True
    ),
    BenchmarkQuestion(
        id="RSN003",
        question="Suggest headphones with noise cancellation for travel",
        category=QuestionCategory.REASONING,
        difficulty=Difficulty.MEDIUM,
        ground_truth="Sony WH-1000XM5 or Bose QuietComfort Ultra are excellent for travel noise cancellation.",
        expected_entities=["Sony", "Bose"],
        expected_keywords=["headphones", "noise cancellation", "travel"],
        requires_graph=True
    ),
    BenchmarkQuestion(
        id="RSN004",
        question="Which Nike shoe is best for running?",
        category=QuestionCategory.REASONING,
        difficulty=Difficulty.HARD,
        ground_truth="Nike Pegasus 40 or Air Max 270 are popular choices for running.",
        expected_entities=["Nike", "Pegasus", "Air Max"],
        expected_keywords=["running", "shoe", "Nike"],
        requires_graph=True
    ),
    BenchmarkQuestion(
        id="RSN005",
        question="I need durable jeans. What brand to buy?",
        category=QuestionCategory.REASONING,
        difficulty=Difficulty.MEDIUM,
        ground_truth="Levi's is the most renowned brand for durable denim jeans.",
        expected_entities=["Levi's"],
        expected_keywords=["jeans", "durable", "Levi's"],
        requires_graph=True
    ),
    
    # =========================================================================
    # CATEGORY 6: EDGE CASES
    # =========================================================================
    BenchmarkQuestion(
        id="EDGE001",
        question="What is the price of the XYZ-9999 product?",
        category=QuestionCategory.ENTITY_LOOKUP,
        difficulty=Difficulty.HARD,
        ground_truth="There is no product named XYZ-9999 in the database.",
        expected_entities=[],
        expected_keywords=["not found", "no product"],
        requires_graph=False
    ),
    BenchmarkQuestion(
        id="EDGE002",
        question="What smartphone does Adidas make?",
        category=QuestionCategory.RELATIONSHIP,
        difficulty=Difficulty.MEDIUM,
        ground_truth="Adidas does not manufacture smartphones. It is a footwear/clothing brand.",
        expected_entities=["Adidas"],
        expected_keywords=["does not", "footwear"],
        requires_graph=True
    ),
    BenchmarkQuestion(
        id="EDGE003",
        question="What cars does Samsung sell?",
        category=QuestionCategory.RELATIONSHIP,
        difficulty=Difficulty.MEDIUM,
        ground_truth="Samsung does not sell cars in this database (Electronics/Appliances only).",
        expected_entities=["Samsung"],
        expected_keywords=["does not", "cars"],
        requires_graph=True
    ),
]


def get_benchmark_questions(
    category: QuestionCategory = None,
    difficulty: Difficulty = None,
    requires_graph: bool = None,
    limit: int = None
) -> List[BenchmarkQuestion]:
    """
    Get filtered benchmark questions.
    
    Args:
        category: Filter by question category
        difficulty: Filter by difficulty level
        requires_graph: Filter by graph requirement
        limit: Maximum number of questions to return
        
    Returns:
        List of matching BenchmarkQuestion objects
    """
    questions = BENCHMARK_QUESTIONS.copy()
    
    if category:
        questions = [q for q in questions if q.category == category]
    
    if difficulty:
        questions = [q for q in questions if q.difficulty == difficulty]
    
    if requires_graph is not None:
        questions = [q for q in questions if q.requires_graph == requires_graph]
    
    if limit:
        questions = questions[:limit]
    
    return questions


def get_question_by_id(question_id: str) -> BenchmarkQuestion:
    """Get a specific question by ID."""
    for q in BENCHMARK_QUESTIONS:
        if q.id == question_id:
            return q
    return None


def get_statistics() -> Dict[str, Any]:
    """Get statistics about the benchmark dataset."""
    categories = {}
    difficulties = {}
    graph_required = 0
    
    for q in BENCHMARK_QUESTIONS:
        cat = q.category.value
        diff = q.difficulty.value
        
        categories[cat] = categories.get(cat, 0) + 1
        difficulties[diff] = difficulties.get(diff, 0) + 1
        
        if q.requires_graph:
            graph_required += 1
    
    return {
        "total_questions": len(BENCHMARK_QUESTIONS),
        "by_category": categories,
        "by_difficulty": difficulties,
        "graph_required": graph_required,
        "graph_optional": len(BENCHMARK_QUESTIONS) - graph_required
    }
