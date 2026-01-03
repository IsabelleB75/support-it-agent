"""
Tests pour l'API FastAPI Support IT Agent
Tests unitaires qui ne necessitent pas les fichiers modeles
"""
import pytest
import re


# Keywords (copie depuis app.py pour tests independants)
keywords = {
    'network': r'\b(network|wifi|vpn|connect|internet|lan|router)\b',
    'printer': r'\b(printer|imprimante|print|scan|scanner)\b',
    'security': r'\b(security|securite|password|login|access|breach|hack|malware)\b',
    'hardware': r'\b(hardware|laptop|pc|macbook|screen|disk|ssd|cpu)\b',
    'software': r'\b(software|app|update|bug|crash|install|version)\b',
}


def extract_num_features(text):
    """Fonction de test independante"""
    text_lower = text.lower()
    body_len = len(text)
    has_feats = [1 if re.search(regex, text_lower) else 0 for regex in keywords.values()]
    return [body_len, body_len * 0.8, 0.8] + has_feats


def test_extract_num_features():
    """Test extraction des features numeriques"""
    text = "My VPN connection is not working on my laptop"
    features = extract_num_features(text)

    # Verifie qu'on a bien 8 features (body_len, answer_len, ratio + 5 has_*)
    assert len(features) == 8

    # body_length doit etre > 0
    assert features[0] > 0

    # has_network doit etre 1 (VPN est dans keywords network)
    assert features[3] == 1  # has_network

    # has_hardware doit etre 1 (laptop est dans keywords hardware)
    assert features[6] == 1  # has_hardware


def test_extract_num_features_printer():
    """Test detection keyword printer"""
    text = "My printer is not printing"
    features = extract_num_features(text)

    # has_printer doit etre 1
    assert features[4] == 1  # has_printer


def test_extract_num_features_security():
    """Test detection keyword security"""
    text = "I forgot my password and cannot login"
    features = extract_num_features(text)

    # has_security doit etre 1 (password et login sont dans keywords security)
    assert features[5] == 1  # has_security


def test_keywords_defined():
    """Test que les keywords sont bien definis"""
    assert 'network' in keywords
    assert 'printer' in keywords
    assert 'security' in keywords
    assert 'hardware' in keywords
    assert 'software' in keywords


def test_keywords_regex_valid():
    """Test que les regex sont valides"""
    for name, pattern in keywords.items():
        # Verifie que le pattern compile sans erreur
        compiled = re.compile(pattern)
        assert compiled is not None


def test_network_keywords():
    """Test detection mots cles network"""
    test_cases = [
        ("VPN not working", True),
        ("WiFi connection issue", True),
        ("Internet is slow", True),
        ("My screen is broken", False),
    ]
    for text, expected in test_cases:
        result = 1 if re.search(keywords['network'], text.lower()) else 0
        assert result == (1 if expected else 0), f"Failed for: {text}"


def test_software_keywords():
    """Test detection mots cles software"""
    test_cases = [
        ("Cannot install Office", True),
        ("App keeps crashing", True),
        ("Need to update Windows", True),
        ("My keyboard is broken", False),
    ]
    for text, expected in test_cases:
        result = 1 if re.search(keywords['software'], text.lower()) else 0
        assert result == (1 if expected else 0), f"Failed for: {text}"
