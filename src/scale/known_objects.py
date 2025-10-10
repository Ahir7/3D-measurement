"""
Known objects database for scale recovery.

Common objects with known dimensions that can be used as scale references.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class KnownObject:
    """Known object with dimensions."""
    name: str
    width: float  # meters
    height: float  # meters
    depth: Optional[float] = None  # meters
    confidence: float = 0.8  # How confident we are in these dimensions
    aliases: List[str] = None  # Alternative names
    description: str = ""
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []


# Database of common objects with standard dimensions
KNOWN_OBJECTS_DATABASE: Dict[str, KnownObject] = {
    # Paper and Cards
    'credit_card': KnownObject(
        name='credit_card',
        width=0.0856,
        height=0.0540,
        depth=0.0008,
        confidence=0.95,
        aliases=['card', 'bank_card', 'credit_card', 'debit_card'],
        description='Standard ISO/IEC 7810 ID-1 card (85.6 x 53.98 mm)'
    ),
    'a4_paper': KnownObject(
        name='a4_paper',
        width=0.297,
        height=0.210,
        depth=0.0001,
        confidence=0.95,
        aliases=['a4', 'paper', 'a4_sheet'],
        description='A4 paper (297 x 210 mm)'
    ),
    'us_letter': KnownObject(
        name='us_letter',
        width=0.279,
        height=0.216,
        depth=0.0001,
        confidence=0.95,
        aliases=['letter', 'letter_paper', 'us_letter_paper'],
        description='US Letter paper (8.5 x 11 inches)'
    ),
    'business_card': KnownObject(
        name='business_card',
        width=0.089,
        height=0.051,
        depth=0.0004,
        confidence=0.90,
        aliases=['card', 'visiting_card'],
        description='Standard business card (89 x 51 mm)'
    ),
    
    # Electronics
    'smartphone': KnownObject(
        name='smartphone',
        width=0.160,
        height=0.078,
        depth=0.009,
        confidence=0.70,
        aliases=['phone', 'mobile', 'cellphone'],
        description='Average smartphone (160 x 78 x 9 mm)'
    ),
    'iphone_13': KnownObject(
        name='iphone_13',
        width=0.1467,
        height=0.0715,
        depth=0.0077,
        confidence=0.95,
        aliases=['iphone'],
        description='iPhone 13 (146.7 x 71.5 x 7.7 mm)'
    ),
    'keyboard_standard': KnownObject(
        name='keyboard_standard',
        width=0.450,
        height=0.150,
        depth=0.030,
        confidence=0.70,
        aliases=['keyboard', 'computer_keyboard'],
        description='Standard full-size keyboard (~450 x 150 mm)'
    ),
    'laptop_15inch': KnownObject(
        name='laptop_15inch',
        width=0.380,
        height=0.260,
        depth=0.020,
        confidence=0.75,
        aliases=['laptop', 'notebook'],
        description='15-inch laptop (approximate)'
    ),
    
    # Monitors
    'monitor_24inch': KnownObject(
        name='monitor_24inch',
        width=0.531,
        height=0.299,
        depth=0.050,
        confidence=0.75,
        aliases=['monitor', '24_inch_monitor'],
        description='24-inch monitor (16:9 aspect ratio)'
    ),
    'monitor_27inch': KnownObject(
        name='monitor_27inch',
        width=0.597,
        height=0.336,
        depth=0.050,
        confidence=0.75,
        aliases=['27_inch_monitor'],
        description='27-inch monitor (16:9 aspect ratio)'
    ),
    
    # Books and Media
    'book_standard': KnownObject(
        name='book_standard',
        width=0.230,
        height=0.153,
        depth=0.025,
        confidence=0.60,
        aliases=['book', 'paperback'],
        description='Standard paperback book (~230 x 153 mm)'
    ),
    'cd_case': KnownObject(
        name='cd_case',
        width=0.142,
        height=0.125,
        depth=0.010,
        confidence=0.90,
        aliases=['cd', 'cd_jewel_case'],
        description='Standard CD jewel case (142 x 125 mm)'
    ),
    'dvd_case': KnownObject(
        name='dvd_case',
        width=0.190,
        height=0.135,
        depth=0.014,
        confidence=0.90,
        aliases=['dvd', 'dvd_case'],
        description='Standard DVD case (190 x 135 mm)'
    ),
    
    # Office Items
    'pen': KnownObject(
        name='pen',
        width=0.140,
        height=0.010,
        depth=0.010,
        confidence=0.60,
        aliases=['ballpoint', 'pen'],
        description='Standard ballpoint pen (~140 mm)'
    ),
    'pencil': KnownObject(
        name='pencil',
        width=0.190,
        height=0.007,
        depth=0.007,
        confidence=0.65,
        aliases=['pencil'],
        description='Standard pencil (~190 mm)'
    ),
    'ruler_30cm': KnownObject(
        name='ruler_30cm',
        width=0.300,
        height=0.030,
        depth=0.002,
        confidence=0.95,
        aliases=['ruler', '12_inch_ruler'],
        description='30 cm / 12 inch ruler'
    ),
    
    # Household Items
    'can_soda': KnownObject(
        name='can_soda',
        width=0.066,
        height=0.123,
        depth=0.066,
        confidence=0.85,
        aliases=['soda_can', 'can', 'beverage_can'],
        description='Standard 12 oz soda can (66 x 123 mm)'
    ),
    'bottle_water': KnownObject(
        name='bottle_water',
        width=0.065,
        height=0.210,
        depth=0.065,
        confidence=0.75,
        aliases=['water_bottle', 'bottle'],
        description='Standard 500ml water bottle'
    ),
    'mug': KnownObject(
        name='mug',
        width=0.090,
        height=0.100,
        depth=0.090,
        confidence=0.60,
        aliases=['coffee_mug', 'cup'],
        description='Standard coffee mug (~90 mm diameter)'
    ),
    
    # Sports Equipment
    'basketball': KnownObject(
        name='basketball',
        width=0.239,
        height=0.239,
        depth=0.239,
        confidence=0.95,
        aliases=['ball', 'basketball'],
        description='Standard basketball (239 mm diameter)'
    ),
    'tennis_ball': KnownObject(
        name='tennis_ball',
        width=0.067,
        height=0.067,
        depth=0.067,
        confidence=0.95,
        aliases=['ball', 'tennis_ball'],
        description='Tennis ball (67 mm diameter)'
    ),
    
    # Furniture (approximate)
    'desk_standard': KnownObject(
        name='desk_standard',
        width=1.400,
        height=0.750,
        depth=0.600,
        confidence=0.50,
        aliases=['desk', 'table'],
        description='Standard office desk (~140 x 75 cm)'
    ),
    'chair_office': KnownObject(
        name='chair_office',
        width=0.650,
        height=0.900,
        depth=0.650,
        confidence=0.50,
        aliases=['chair', 'office_chair'],
        description='Standard office chair (~65 x 90 cm)'
    ),
}


def get_object_by_name(name: str) -> Optional[KnownObject]:
    """
    Get known object by name or alias.
    
    Args:
        name: Object name or alias
        
    Returns:
        KnownObject if found, None otherwise
    """
    name_lower = name.lower().strip()
    
    # Direct match
    if name_lower in KNOWN_OBJECTS_DATABASE:
        return KNOWN_OBJECTS_DATABASE[name_lower]
    
    # Check aliases
    for obj in KNOWN_OBJECTS_DATABASE.values():
        if name_lower in [alias.lower() for alias in obj.aliases]:
            return obj
    
    return None


def get_all_object_names() -> List[str]:
    """Get list of all known object names."""
    return list(KNOWN_OBJECTS_DATABASE.keys())


def get_objects_by_confidence(min_confidence: float = 0.8) -> List[KnownObject]:
    """
    Get objects with at least the specified confidence.
    
    Args:
        min_confidence: Minimum confidence threshold
        
    Returns:
        List of high-confidence objects
    """
    return [obj for obj in KNOWN_OBJECTS_DATABASE.values() 
            if obj.confidence >= min_confidence]


def suggest_objects_for_scale(scene_type: str = 'indoor') -> List[str]:
    """
    Suggest good objects to use for scale based on scene type.
    
    Args:
        scene_type: 'indoor', 'outdoor', 'desktop', or 'general'
        
    Returns:
        List of recommended object names
    """
    recommendations = {
        'desktop': [
            'credit_card', 'a4_paper', 'business_card', 'keyboard_standard',
            'smartphone', 'pen', 'ruler_30cm', 'cd_case', 'book_standard'
        ],
        'indoor': [
            'a4_paper', 'book_standard', 'laptop_15inch', 'monitor_24inch',
            'chair_office', 'desk_standard', 'bottle_water'
        ],
        'outdoor': [
            'smartphone', 'bottle_water', 'basketball', 'tennis_ball'
        ],
        'general': [
            'credit_card', 'a4_paper', 'smartphone', 'bottle_water',
            'keyboard_standard', 'ruler_30cm'
        ]
    }
    
    return recommendations.get(scene_type, recommendations['general'])


def print_known_objects_guide():
    """Print a user-friendly guide of known objects."""
    print("\n" + "=" * 70)
    print("KNOWN OBJECTS FOR SCALE REFERENCE")
    print("=" * 70)
    print("\nPlace one of these objects in your scene for accurate scale:")
    print()
    
    categories = {
        'High Confidence (Recommended)': [obj for obj in KNOWN_OBJECTS_DATABASE.values() if obj.confidence >= 0.90],
        'Good Confidence': [obj for obj in KNOWN_OBJECTS_DATABASE.values() if 0.75 <= obj.confidence < 0.90],
        'Fair Confidence': [obj for obj in KNOWN_OBJECTS_DATABASE.values() if obj.confidence < 0.75],
    }
    
    for category, objects in categories.items():
        if objects:
            print(f"{category}:")
            print("-" * 70)
            for obj in sorted(objects, key=lambda x: x.confidence, reverse=True):
                dims = f"{obj.width*100:.1f} x {obj.height*100:.1f}"
                if obj.depth:
                    dims += f" x {obj.depth*100:.1f}"
                print(f"  • {obj.name:25s} {dims:20s} cm  ({obj.confidence:.0%})")
            print()
    
    print("=" * 70)
    print("\nTIP: Use ArUco markers for even better accuracy!")
    print()


if __name__ == "__main__":
    # Print guide when run directly
    print_known_objects_guide()
    
    # Test lookups
    print("\nTest Lookups:")
    print(f"credit_card: {get_object_by_name('credit_card')}")
    print(f"smartphone: {get_object_by_name('smartphone')}")
    print(f"\nDesktop scene suggestions: {suggest_objects_for_scale('desktop')}")

