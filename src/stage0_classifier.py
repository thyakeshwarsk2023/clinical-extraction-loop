import re

class AssayClassifier:
    DIRECT_PATTERNS = [
        r"\bdirect\b",
        r"\bhomogeneous\b",
        r"\bimmunoinhibition\b",
        r"\benzymatic\s+clearance\b",
        r"\bldl-c\s+direct\b"
    ]
    
    CALCULATED_PATTERNS = [
        r"\bfriedewald\b",
        r"\bcalculated\b",
        r"\bcalc\b",
        r"\bderived\b",
        r"\bcholesterol\s*-\s*hdl\s*-\s*vldl\b"
    ]
    
    MARTIN_PATTERNS = [
        r"\bmartin[- ]hopkins\b",
        r"\badjustable\s+factor\b",
        r"\bnovel\s+ldl\b"
    ]

    @classmethod
    def classify(cls, document_text: str) -> dict:
        text = document_text.lower()
        
        # Check Direct first (highest clinical priority to prevent false triggers)
        for pattern in cls.DIRECT_PATTERNS:
            if re.search(pattern, text):
                return {
                    "route": "DIRECT_MEASURED",
                    "matched_pattern": pattern,
                    "confidence": 1.0,
                    "apply_equality_invariant": False
                }
                
        # Check Martin-Hopkins
        for pattern in cls.MARTIN_PATTERNS:
            if re.search(pattern, text):
                return {
                    "route": "MARTIN_HOPKINS",
                    "matched_pattern": pattern,
                    "confidence": 1.0,
                    "apply_equality_invariant": True
                }
                
        # Check Friedewald
        for pattern in cls.CALCULATED_PATTERNS:
            if re.search(pattern, text):
                return {
                    "route": "FRIEDEWALD",
                    "matched_pattern": pattern,
                    "confidence": 1.0,
                    "apply_equality_invariant": True
                }
                
        # First-class ambiguous outcome
        return {
            "route": "AMBIGUOUS_FALLBACK",
            "matched_pattern": None,
            "confidence": 0.0,
            "apply_equality_invariant": False
        }