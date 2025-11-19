from datetime import datetime
from typing import Optional

def trimestre_to_date_fin(trim_str: str) -> Optional[datetime]:
    """
    Convertit un string de format 'AAAA-TX' en une date de FIN du trimestre.
    
    Args:
        trim_str (str): String au format 'AAAA-TX' (ex: '2020-T1')
    
    Returns:
        datetime or None: Date du dernier jour du trimestre ou None si conversion impossible
    """
    
        # Nettoyer la string (enlever les espaces, etc.)
    trim_str = str(trim_str).strip()
        
        # Vérifier le format basique
    if '-' not in trim_str or len(trim_str) < 6:
        return None
            
        # Extraire année et trimestre
    annee_part = trim_str.split('-')[0]  # '2020'
    trimestre_part = trim_str.split('-')[1]  # 'T1'
        
        # Garder seulement les chiffres
    annee = ''.join(filter(str.isdigit, annee_part))
    trimestre = ''.join(filter(str.isdigit, trimestre_part))
        
    if not annee or not trimestre:
        return None
            
    annee = int(annee)
    trimestre = int(trimestre)
        
    if not (1 <= trimestre <= 4):
        return None
            
        # Date de FIN du trimestre
    if trimestre == 1:    
            return datetime(annee, 3, 31)
    elif trimestre == 2: 
            return datetime(annee, 6, 30)
    elif trimestre == 3: 
            return datetime(annee, 9, 30)
    else:                 
            return datetime(annee, 12, 31)