"""
Knowledge Graph for E-commerce Products

Builds and manages relationships between:
- Products
- Brands  
- Categories
- Price Ranges
- Features
"""
import json
import logging
from typing import Dict, List, Set, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """
    A knowledge graph that stores entities and their relationships.
    
    Entity Types:
    - product: Individual products
    - brand: Product brands
    - category: Product categories
    - price_range: Price brackets (budget, mid-range, premium)
    - feature: Product features
    
    Relationship Types:
    - MADE_BY: Product -> Brand
    - BELONGS_TO: Product -> Category
    - IN_PRICE_RANGE: Product -> Price Range
    - HAS_FEATURE: Product -> Feature
    - SIMILAR_TO: Product -> Product
    """
    
    def __init__(self):
        # Entities: {entity_id: {type, name, properties}}
        self.entities: Dict[str, Dict[str, Any]] = {}
        
        # Relationships: {source_id: [(relationship_type, target_id)]}
        self.relationships: Dict[str, List[tuple]] = defaultdict(list)
        
        # Reverse index for efficient lookups
        self.reverse_relationships: Dict[str, List[tuple]] = defaultdict(list)
        
        # Entity type index
        self.entities_by_type: Dict[str, Set[str]] = defaultdict(set)
        
        # Name to ID mapping
        self.name_to_id: Dict[str, str] = {}
        
        logger.info("Knowledge Graph initialized")
    
    def _generate_id(self, entity_type: str, name: str) -> str:
        """Generate a unique ID for an entity."""
        clean_name = name.lower().replace(" ", "_").replace("'", "").replace("-", "_")
        return f"{entity_type}:{clean_name}"
    
    def add_entity(self, entity_type: str, name: str, properties: Optional[Dict] = None) -> str:
        """
        Add an entity to the graph.
        
        Args:
            entity_type: Type of entity (product, brand, category, etc.)
            name: Name of the entity
            properties: Additional properties
            
        Returns:
            Entity ID
        """
        entity_id = self._generate_id(entity_type, name)
        
        if entity_id not in self.entities:
            self.entities[entity_id] = {
                "id": entity_id,
                "type": entity_type,
                "name": name,
                "properties": properties or {}
            }
            self.entities_by_type[entity_type].add(entity_id)
            self.name_to_id[name.lower()] = entity_id
        else:
            # Update properties if entity already exists
            if properties:
                self.entities[entity_id]["properties"].update(properties)
        
        return entity_id
    
    def add_relationship(self, source_id: str, relationship_type: str, target_id: str):
        """
        Add a relationship between two entities.
        
        Args:
            source_id: Source entity ID
            relationship_type: Type of relationship (MADE_BY, BELONGS_TO, etc.)
            target_id: Target entity ID
        """
        if source_id not in self.entities or target_id not in self.entities:
            logger.warning(f"Cannot add relationship: entity not found")
            return
        
        rel_tuple = (relationship_type, target_id)
        if rel_tuple not in self.relationships[source_id]:
            self.relationships[source_id].append(rel_tuple)
            self.reverse_relationships[target_id].append((relationship_type, source_id))
    
    def get_entity(self, entity_id: str) -> Optional[Dict]:
        """Get an entity by ID."""
        return self.entities.get(entity_id)
    
    def find_entity(self, name: str) -> Optional[Dict]:
        """Find an entity by name."""
        entity_id = self.name_to_id.get(name.lower())
        if entity_id:
            return self.entities.get(entity_id)
        return None
    
    def search_entities(self, query: str, entity_type: Optional[str] = None, limit: int = 5) -> List[Dict]:
        """
        Search for entities in the graph matching the query.
        Uses precision-focused token overlap matching.
        """
        import re
        
        def get_tokens(text):
            return set(re.findall(r'\w+', text.lower()))

        query_tokens = get_tokens(query)
        # Filter short/common tokens (stop words)
        stop_words = {'the', 'is', 'are', 'what', 'which', 'how', 'does', 'for', 'and', 'or', 'any', 'all', 'has', 'have', 'can', 'with', 'from', 'this', 'that', 'there', 'products', 'product'}
        valid_tokens = {t for t in query_tokens if len(t) > 2 and t not in stop_words}
        
        if not valid_tokens:
            return []
        
        scored_matches = []
        seen_ids = set()
        
        for name, entity_id in self.name_to_id.items():
            if entity_type:
                entity = self.entities[entity_id]
                if entity["type"] != entity_type:
                    continue

            name_tokens = get_tokens(name)
            if not name_tokens:
                continue
            
            common = valid_tokens.intersection(name_tokens)
            
            if common:
                # Precision: what fraction of entity name tokens matched?
                name_precision = len(common) / len(name_tokens)
                # Recall: what fraction of query tokens matched?
                query_recall = len(common) / len(valid_tokens)
                
                # Require at least 50% of the entity name tokens to match
                # This prevents "apple" matching "apple oppo a 66" (only 1/4 tokens)
                if name_precision >= 0.5 or len(common) >= 2:
                    if entity_id not in seen_ids:
                        score = name_precision * 0.6 + query_recall * 0.4
                        scored_matches.append((score, self.entities[entity_id]))
                        seen_ids.add(entity_id)
        
        # Sort by score descending
        scored_matches.sort(key=lambda x: x[0], reverse=True)
        
        return [m[1] for m in scored_matches[:limit]]

    def get_related_entities(self, entity_id: str) -> List[Dict]:
        """Get related entities (alias for get_related)."""
        return self.get_related(entity_id)

    def get_related(self, entity_id: str, relationship_type: Optional[str] = None) -> List[Dict]:
        """
        Get entities related to a given entity.
        
        Args:
            entity_id: Source entity ID
            relationship_type: Filter by relationship type (optional)
            
        Returns:
            List of related entities with relationship info
        """
        related = []
        
        # Outgoing relationships
        if entity_id in self.relationships:
            for rel_type, target_id in self.relationships[entity_id]:
                if relationship_type is None or rel_type == relationship_type:
                    entity = self.entities.get(target_id)
                    if entity:
                        related.append({
                            "entity": entity,
                            "relationship": rel_type,
                            "direction": "outgoing"
                        })
        
        # Incoming relationships
        if entity_id in self.reverse_relationships:
            for rel_type, source_id in self.reverse_relationships[entity_id]:
                if relationship_type is None or rel_type == relationship_type:
                    entity = self.entities.get(source_id)
                    if entity:
                        related.append({
                            "entity": entity,

                        "relationship": rel_type,
                        "direction": "incoming"
                    })
        
        return related
    
    def get_entities_by_type(self, entity_type: str) -> List[Dict]:
        """Get all entities of a specific type."""
        return [
            self.entities[eid] 
            for eid in self.entities_by_type.get(entity_type, set())
        ]
    
    def get_products_by_brand(self, brand_name: str) -> List[Dict]:
        """Get all products for a specific brand."""
        brand_id = self._generate_id("brand", brand_name)
        products = []
        
        for rel_type, source_id in self.reverse_relationships.get(brand_id, []):
            if rel_type == "MADE_BY":
                product = self.entities.get(source_id)
                if product:
                    products.append(product)
        
        return products
    
    def get_products_by_category(self, category_name: str) -> List[Dict]:
        """Get all products in a specific category."""
        category_id = self._generate_id("category", category_name)
        products = []
        
        for rel_type, source_id in self.reverse_relationships.get(category_id, []):
            if rel_type == "BELONGS_TO":
                product = self.entities.get(source_id)
                if product:
                    products.append(product)
        
        return products
    

    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        total_relationships = sum(len(rels) for rels in self.relationships.values())
        
        return {
            "total_entities": len(self.entities),
            "total_relationships": total_relationships,
            "entities_by_type": {
                etype: len(eids) 
                for etype, eids in self.entities_by_type.items()
            },
            "top_brands": self._get_top_entities("brand", 10),
            "top_categories": self._get_top_entities("category", 10)
        }
    
    def _get_top_entities(self, entity_type: str, limit: int) -> List[Dict]:
        """Get entities with most relationships."""
        entities = []
        for entity_id in self.entities_by_type.get(entity_type, set()):
            entity = self.entities[entity_id]
            count = len(self.reverse_relationships.get(entity_id, []))
            entities.append({"name": entity["name"], "count": count})
        
        return sorted(entities, key=lambda x: -x["count"])[:limit]
    
    def to_d3_format(self, max_nodes: int = 200) -> Dict[str, Any]:
        """
        Convert graph to D3.js force-directed graph format.
        
        Returns:
            {nodes: [...], links: [...]}
        """
        nodes = []
        links = []
        node_ids = set()
        
        # Sample entities if too many
        all_entity_ids = list(self.entities.keys())
        if len(all_entity_ids) > max_nodes:
            # Prioritize brands and categories, then sample products
            priority_ids = []
            for etype in ["brand", "category", "price_range"]:
                priority_ids.extend(list(self.entities_by_type.get(etype, set())))
            
            remaining = max_nodes - len(priority_ids)
            product_ids = list(self.entities_by_type.get("product", set()))[:remaining]
            selected_ids = set(priority_ids + product_ids)
        else:
            selected_ids = set(all_entity_ids)
        
        # Build nodes
        for entity_id in selected_ids:
            entity = self.entities[entity_id]
            nodes.append({
                "id": entity_id,
                "name": entity["name"],
                "type": entity["type"],
                "group": self._get_group_number(entity["type"])
            })
            node_ids.add(entity_id)
        
        # Build links (only for selected nodes)
        for source_id in selected_ids:
            for rel_type, target_id in self.relationships.get(source_id, []):
                if target_id in node_ids:
                    links.append({
                        "source": source_id,
                        "target": target_id,
                        "type": rel_type
                    })
        
        return {"nodes": nodes, "links": links}
    
    def _get_group_number(self, entity_type: str) -> int:
        """Get group number for D3 visualization coloring."""
        groups = {
            "product": 1,
            "brand": 2,
            "category": 3,
            "price_range": 4,
            "feature": 5
        }
        return groups.get(entity_type, 0)
    
    def build_from_products(self, products: List[Dict[str, Any]]):
        """
        Build the knowledge graph from product data.
        
        Args:
            products: List of product dictionaries
        """
        logger.info(f"Building knowledge graph from {len(products)} products...")
        
        # Define price ranges
        price_ranges = [
            ("Budget (Under ₹500)", 0, 500),
            ("Affordable (₹500-₹1000)", 500, 1000),
            ("Mid-Range (₹1000-₹2500)", 1000, 2500),
            ("Premium (₹2500-₹5000)", 2500, 5000),
            ("Luxury (₹5000-₹10000)", 5000, 10000),
            ("Ultra Premium (Above ₹10000)", 10000, float('inf'))
        ]
        
        # Add price range entities
        for pr_name, _, _ in price_ranges:
            self.add_entity("price_range", pr_name)
        
        for product in products:
            # Add product entity with rich properties
            product_name = product.get("name", "Unknown")
            product_id = self.add_entity("product", product_name, {
                "price": product.get("price"),
                "rating": product.get("rating"),
                "reviews_count": product.get("reviews_count"),
                "brand": product.get("brand", ""),
                "category": product.get("category", ""),
                "features": product.get("features", [])[:5],  # Limit to 5 features
                "description": product.get("description", "")[:200] if product.get("description") else ""  # Truncate description
            })
            
            # Add brand and create relationship
            brand = product.get("brand")
            if brand:
                brand_id = self.add_entity("brand", brand)
                self.add_relationship(product_id, "MADE_BY", brand_id)
            
            # Add category and create relationship
            category = product.get("category", "")
            if category:
                # Handle compound categories like "Kitchen - Pressure Cooker"
                main_category = category.split(" - ")[0] if " - " in category else category
                category_id = self.add_entity("category", main_category)
                self.add_relationship(product_id, "BELONGS_TO", category_id)
            
            # Add price range relationship
            price = product.get("price", 0)
            for pr_name, pr_min, pr_max in price_ranges:
                if pr_min <= price < pr_max:
                    pr_id = self._generate_id("price_range", pr_name)
                    self.add_relationship(product_id, "IN_PRICE_RANGE", pr_id)
                    break
            
            # Add feature relationships
            features = product.get("features", [])
            for feature in features[:3]:  # Limit features per product
                feature_id = self.add_entity("feature", feature)
                self.add_relationship(product_id, "HAS_FEATURE", feature_id)
        
        logger.info(f"Knowledge graph built: {len(self.entities)} entities, "
                    f"{sum(len(r) for r in self.relationships.values())} relationships")
    
    def to_json(self) -> str:
        """Export graph as JSON."""
        return json.dumps({
            "entities": self.entities,
            "relationships": dict(self.relationships),
            "stats": self.get_stats()
        }, indent=2)


# Global knowledge graph instance
_knowledge_graph: Optional[KnowledgeGraph] = None


def get_knowledge_graph() -> KnowledgeGraph:
    """Get or create the global knowledge graph instance."""
    global _knowledge_graph
    if _knowledge_graph is None:
        _knowledge_graph = KnowledgeGraph()
    return _knowledge_graph


def reset_knowledge_graph():
    """Reset the knowledge graph."""
    global _knowledge_graph
    _knowledge_graph = KnowledgeGraph()
