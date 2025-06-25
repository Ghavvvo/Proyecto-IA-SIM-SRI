"""
Script de prueba para verificar que las preferencias del último mensaje
se incluyen correctamente al buscar información
"""

from agent_tourist_guide import TouristGuideAgent
import json

def test_preferences_extraction():
    """
    Prueba que las preferencias se extraen correctamente del último mensaje
    """
    print("=== TEST: Extracción de preferencias del último mensaje ===\n")
    
    
    guide = TouristGuideAgent("TestGuide")
    
    
    response = guide.receive({'type': 'start_conversation'}, None)
    print(f"Guía: {response['message']}\n")
    
    
    messages = [
        "Quiero ir a Cuba",
        "Me interesan las playas y los museos",
        "También quiero conocer buenos restaurantes, eso es todo"
    ]
    
    for msg in messages:
        print(f"Usuario: {msg}")
        response = guide.receive({'type': 'user_message', 'message': msg}, None)
        print(f"Guía: {response['message']}")
        
        
        current_prefs = response.get('current_preferences', {})
        print(f"\nPreferencias actuales:")
        print(f"- Destino: {current_prefs.get('destination')}")
        print(f"- Intereses: {current_prefs.get('interests')}")
        print(f"- Recopilación completa: {response.get('preferences_collected', False)}")
        print("-" * 50 + "\n")
    
    
    final_response = guide.receive({'type': 'get_preferences'}, None)
    final_prefs = final_response['preferences']
    
    print("\n=== PREFERENCIAS FINALES ===")
    print(json.dumps(final_prefs, indent=2, ensure_ascii=False))
    
    
    expected_interests = ['beaches', 'museums', 'restaurants']
    actual_interests = final_prefs.get('interests', [])
    
    print("\n=== VERIFICACIÓN ===")
    print(f"Intereses esperados: {expected_interests}")
    print(f"Intereses detectados: {actual_interests}")
    
    
    missing_interests = []
    for interest in expected_interests:
        if interest not in actual_interests:
            missing_interests.append(interest)
    
    if missing_interests:
        print(f"\n❌ ERROR: Faltan los siguientes intereses: {missing_interests}")
        return False
    else:
        print(f"\n✅ ÉXITO: Todos los intereses fueron detectados correctamente")
        return True

def test_preferences_in_coordinator_flow():
    """
    Prueba el flujo completo con el coordinador (simulado)
    """
    print("\n\n=== TEST: Flujo completo con preferencias ===\n")
    
    
    guide = TouristGuideAgent("TestGuide")
    
    
    guide.receive({'type': 'start_conversation'}, None)
    
    
    messages = [
        "Cuba",
        "playas y museos", 
        "restaurantes también, ya"
    ]
    
    last_response = None
    for msg in messages:
        last_response = guide.receive({'type': 'user_message', 'message': msg}, None)
    
    
    print("=== Verificando último response ===")
    print(f"preferences_collected: {last_response.get('preferences_collected')}")
    print(f"final_preferences presente: {'final_preferences' in last_response}")
    print(f"current_preferences presente: {'current_preferences' in last_response}")
    
    if 'current_preferences' in last_response:
        current = last_response['current_preferences']
        print(f"\nPreferencias en current_preferences:")
        print(f"- Destino: {current.get('destination')}")
        print(f"- Intereses: {current.get('interests')}")
    
    if 'final_preferences' in last_response:
        final = last_response['final_preferences']
        print(f"\nPreferencias en final_preferences:")
        print(f"- Destino: {final.get('destination')}")
        print(f"- Intereses: {final.get('interests')}")
    
    
    success = True
    if last_response.get('preferences_collected'):
        prefs = last_response.get('current_preferences') or last_response.get('final_preferences')
        if prefs:
            interests = prefs.get('interests', [])
            if 'restaurants' not in interests:
                print("\n❌ ERROR: 'restaurants' no está en las preferencias finales")
                success = False
            else:
                print("\n✅ ÉXITO: Todas las preferencias están incluidas en el response final")
        else:
            print("\n❌ ERROR: No se encontraron preferencias en el response")
            success = False
    
    return success

if __name__ == "__main__":
    
    test1_passed = test_preferences_extraction()
    test2_passed = test_preferences_in_coordinator_flow()
    
    print("\n\n=== RESUMEN DE PRUEBAS ===")
    print(f"Test 1 (Extracción de preferencias): {'✅ PASÓ' if test1_passed else '❌ FALLÓ'}")
    print(f"Test 2 (Flujo con coordinador): {'✅ PASÓ' if test2_passed else '❌ FALLÓ'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 Todas las pruebas pasaron exitosamente!")
    else:
        print("\n⚠️ Algunas pruebas fallaron. Revisar los logs anteriores.")