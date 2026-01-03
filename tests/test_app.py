"""
Tests pour l'API FastAPI Support IT Agent
"""
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_extract_num_features():
    """Test extraction des features numeriques"""
    from app import extract_num_features

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
    from app import extract_num_features

    text = "My printer is not printing"
    features = extract_num_features(text)

    # has_printer doit etre 1
    assert features[4] == 1  # has_printer


def test_extract_num_features_security():
    """Test detection keyword security"""
    from app import extract_num_features

    text = "I forgot my password and cannot login"
    features = extract_num_features(text)

    # has_security doit etre 1 (password et login sont dans keywords security)
    assert features[5] == 1  # has_security


def test_keywords_defined():
    """Test que les keywords sont bien definis"""
    from app import keywords

    assert 'network' in keywords
    assert 'printer' in keywords
    assert 'security' in keywords
    assert 'hardware' in keywords
    assert 'software' in keywords


def test_api_health():
    """Test endpoint health"""
    from fastapi.testclient import TestClient
    from app import app

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
