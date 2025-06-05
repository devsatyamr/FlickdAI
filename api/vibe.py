from transformers import pipeline
import json
import os

# Load vibe labels from JSON file
base_dir = os.path.dirname(os.path.dirname(__file__))
with open(os.path.join(base_dir, 'vibeslist.json'), 'r') as f:
    VIBE_LABELS = json.load(f)

# Style hint mapping
STYLE_HINTS = {
    'dress': {
        'black': ['Party Glam', 'Clean Girl', 'Streetcore'],
        'white': ['Coquette', 'Clean Girl', 'Cottagecore'],
        'gray': ['Clean Girl', 'Streetcore'],
        'brown': ['Boho', 'Cottagecore'],
        'blue': ['Clean Girl', 'Coquette'],
        'red': ['Party Glam', 'Y2K'],
        'purple': ['Party Glam', 'Y2K'],
        'green': ['Cottagecore', 'Boho'],
        'yellow': ['Y2K', 'Boho'],
        'orange': ['Boho', 'Y2K']
    },
    'top': {
        'black': ['Streetcore', 'Clean Girl', 'Party Glam'],
        'white': ['Clean Girl', 'Coquette', 'Cottagecore'],
        'gray': ['Clean Girl', 'Streetcore'],
        'brown': ['Boho', 'Clean Girl'],
        'blue': ['Clean Girl', 'Coquette'],
        'red': ['Y2K', 'Party Glam'],
        'purple': ['Y2K', 'Party Glam'],
        'green': ['Cottagecore', 'Boho'],
        'yellow': ['Y2K', 'Boho'],
        'orange': ['Boho', 'Y2K']
    },
    'pants': {
        'black': ['Streetcore', 'Clean Girl'],
        'white': ['Clean Girl', 'Coquette'],
        'gray': ['Clean Girl', 'Streetcore'],
        'brown': ['Boho', 'Clean Girl'],
        'blue': ['Clean Girl', 'Streetcore'],
        'red': ['Y2K'],
        'purple': ['Y2K'],
        'green': ['Boho'],
        'yellow': ['Y2K'],
        'orange': ['Boho']
    }
}

class VibeClassifier:
    def __init__(self):
        self.classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

    def _get_description_from_products(self, products):
        """Generate a description from detected products with style hints"""
        if not products:
            return "", []
            
        # Get unique items by type and color
        items = {}
        style_hints = set()
        
        for p in products:
            key = (p['type'], p['color'])
            if key not in items:
                items[key] = 1
            else:
                items[key] += 1
                
            # Add style hints based on item type and color
            item_type = p['type']
            color = p['color']
            if item_type in STYLE_HINTS and color in STYLE_HINTS[item_type]:
                style_hints.update(STYLE_HINTS[item_type][color])
                
        # Generate description
        desc_parts = []
        for (type_, color), count in items.items():
            part = f"{color} {type_}"
            if count > 1:
                part = f"multiple {part}s"
            desc_parts.append(part)
            
        return ", ".join(desc_parts), list(style_hints)

    def classify(self, text="", products=None):
        try:
            # Generate description and get style hints
            product_desc, style_hints = self._get_description_from_products(products)
            
            # If no caption provided, use product description
            if not text.strip():
                if product_desc:
                    base_text = f"An outfit featuring {product_desc}"
                    # Add style hints
                    if style_hints:
                        base_text += f". This combination suggests {', '.join(style_hints)} style elements"
                    text = base_text
                else:
                    return []
            else:
                # Combine caption with product description and style hints
                text = f"{text}. The outfit includes: {product_desc}"
                if style_hints:
                    text += f". Style elements suggest: {', '.join(style_hints)}"
            
            print(f"\nAnalyzing outfit: {text}")
            
            # Run classification with lower threshold
            result = self.classifier(
                text, 
                VIBE_LABELS,
                multi_label=True,
                hypothesis_template="This outfit is in {} style."
            )

            # Get all scores
            all_vibes = list(zip(result['labels'], result['scores']))
            print("\nAll vibe scores:")
            for vibe, score in all_vibes:
                print(f"{vibe}: {score:.3f}")            # Combine BART predictions with style hints
            vibes_with_scores = []
            
            # Add BART predictions with lowered threshold
            vibes_with_scores.extend([
                (label, score) for label, score in all_vibes
                if score > 0.1  # Even lower threshold
            ])
            
            # Add style hints with high confidence
            for hint in style_hints:
                # Check if hint already exists
                existing = [v for v in vibes_with_scores if v[0] == hint]
                if existing:
                    # Take max of existing score and hint score
                    idx = vibes_with_scores.index(existing[0])
                    vibes_with_scores[idx] = (hint, max(existing[0][1], 0.7))
                else:
                    vibes_with_scores.append((hint, 0.7))
            
            # Sort by score and ensure we return at least 2 vibes if possible
            vibes_with_scores.sort(key=lambda x: x[1], reverse=True)
            vibes = [v[0] for v in vibes_with_scores[:3]]
            
            # If we have no vibes yet but have products, force include some basic vibes
            if not vibes and products:
                basic_vibes = ["Clean Girl", "Streetcore"]  # Safe fallback options
                vibes.extend(basic_vibes[:2])
            
            print(f"\nFinal vibes selected: {vibes}")
            return vibes
            
        except Exception as e:
            print(f"Error in vibe classification: {str(e)}")
            return []