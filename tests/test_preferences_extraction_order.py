"""
Script de prueba para verificar que las preferencias se extraen correctamente
incluso cuando el usuario indica que no quiere dar más información en el mismo mensaje
"""

from agent_tourist_guide import TouristGuideAgent
import json

def test_extraction_before_proceed_check():
    """
    Prueba que las preferencias se extraen ANTES de verificar si el usuario quiere proceder
    """
    print("=== TEST: Extracción de preferencias antes de verificar intención de proceder ===\n")
    
    # Crear agente turístico
    guide = TouristGuideAgent("TestGuide")
    
    # Simular estado inicial con destino ya establecido
    guide.conversation_state['phase'] = 'interests'
    guide.conversation_state['preferences']['destination'] = 'Cuba'
    
    # Casos de prueba donde el usuario da información Y dice que no quiere dar más
    test_cases = [
        {
            'message': "Solo me interesan museos, nada más",
            'expected_interest': 'museums',
            'description': 'Usuario especifica interés y dice "nada más"'
        },
        {
            'message': "Quiero ir a playas, eso es todo",
            'expected_interest': 'beaches',
            'description': 'Usuario especifica interés y dice "eso es todo"'
        },
        {
            'message': "Restaurantes y hoteles, ya no preguntes más",
            'expected_interests': ['restaurants', 'accommodation'],
            'description': 'Usuario especifica múltiples intereses y dice "ya no preguntes más"'
        },
        {
            'message': "Me gustan los museos y las playas, suficiente información",
            'expected_interests': ['museums', 'beaches'],
            'description': 'Usuario especifica intereses y dice "suficiente información"'
        }
    ]
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nCaso {i}: {test_case['description']}")
        print(f"Mensaje: '{test_case['message']}'")
        
        # Resetear intereses para cada prueba
        guide.conversation_state['preferences']['interests'] = []
        
        # Simular extracción manual (para evitar llamadas a API)
        guide._manual_extraction(test_case['message'])
        
        # Verificar resultados
        actual_interests = guide.conversation_state['preferences']['interests']
        print(f"Intereses extraídos: {actual_interests}")
        
        # Verificar si se extrajeron los intereses esperados
        if 'expected_interest' in test_case:
            expected = [test_case['expected_interest']]
        else:
            expected = test_case['expected_interests']
        
        success = all(interest in actual_interests for interest in expected)
        
        if success:
            print("✅ ÉXITO: Se extrajeron todos los intereses esperados")
        else:
            print(f"❌ FALLO: Esperados {expected}, obtenidos {actual_interests}")
            all_passed = False
    
    return all_passed

def test_process_flow():
    """
    Prueba el flujo completo de procesamiento de mensajes
    """
    print("\n\n=== TEST: Flujo completo de procesamiento ===\n")
    
    # Crear agente turístico
    guide = TouristGuideAgent("TestGuide")
    
    # Simular estado con destino
    guide.conversation_state['phase'] = 'interests'
    guide.conversation_state['preferences']['destination'] = 'México'
    
    # Mensaje que combina preferencia con indicación de no dar más info
    test_message = "Solo quiero conocer museos, nada más"
    
    print(f"Estado inicial:")
    print(f"- Destino: {guide.conversation_state['preferences']['destination']}")
    print(f"- Intereses: {guide.conversation_state['preferences']['interests']}")
    print(f"\nProcesando mensaje: '{test_message}'")
    
    # Simular el flujo de _process_user_message
    # 1. Agregar al historial
    guide.conversation_state['conversation_history'].append({
        'role': 'user',
        'content': test_message,
        'timestamp': 'test'
    })
    
    # 2. Extraer preferencias (esto debe ocurrir ANTES de verificar si quiere proceder)
    guide._manual_extraction(test_message)
    
    print(f"\nDespués de extracción:")
    print(f"- Intereses: {guide.conversation_state['preferences']['interests']}")
    
    # 3. Verificar si quiere proceder
    wants_to_proceed = guide._wants_to_proceed_with_current_info(test_message)
    print(f"\n¿Usuario quiere proceder?: {wants_to_proceed}")
    
    # Verificar que se extrajo el interés antes de proceder
    if 'museums' in guide.conversation_state['preferences']['interests'] and wants_to_proceed:
        print("\n✅ ÉXITO: Se extrajo el interés 'museums' antes de detectar que el usuario quiere proceder")
        return True
    else:
        print("\n❌ FALLO: No se extrajo el interés correctamente")
        return False

def test_order_of_operations():
    """
    Verifica que el orden de operaciones en _process_user_message es correcto
    """
    print("\n\n=== TEST: Orden de operaciones ===\n")
    
    # Leer el código del método para verificar el orden
    import inspect
    
    guide = TouristGuideAgent("TestGuide")
    source_lines = inspect.getsource(guide._process_user_message).split('\n')
    
    # Buscar las líneas clave
    extract_line = -1
    proceed_line = -1
    
    for i, line in enumerate(source_lines):
        if '_extract_preferences' in line:
            extract_line = i
        elif '_wants_to_proceed_with_current_info' in line:
            proceed_line = i
    
    print(f"Línea donde se llama _extract_preferences: {extract_line}")
    print(f"Línea donde se llama _wants_to_proceed_with_current_info: {proceed_line}")
    
    if extract_line > 0 and proceed_line > 0 and extract_line < proceed_line:
        print("\n✅ ÉXITO: La extracción de preferencias ocurre ANTES de verificar si el usuario quiere proceder")
        return True
    else:
        print("\n❌ FALLO: El orden de operaciones no es correcto")
        return False

if __name__ == "__main__":
    # Ejecutar pruebas
    test1_passed = test_extraction_before_proceed_check()
    test2_passed = test_process_flow()
    test3_passed = test_order_of_operations()
    
    print("\n\n=== RESUMEN DE PRUEBAS ===")
    print(f"Test 1 (Extracción en mensajes combinados): {'✅ PASÓ' if test1_passed else '❌ FALLÓ'}")
    print(f"Test 2 (Flujo completo): {'✅ PASÓ' if test2_passed else '❌ FALLÓ'}")
    print(f"Test 3 (Orden de operaciones): {'✅ PASÓ' if test3_passed else '❌ FALLÓ'}")
    
    if test1_passed and test2_passed and test3_passed:
        print("\n🎉 Todas las pruebas pasaron exitosamente!")
        print("\nLa corrección implementada asegura que:")
        print("1. Las preferencias se extraen ANTES de verificar si el usuario quiere proceder")
        print("2. Mensajes como 'Solo me interesan museos, nada más' extraen correctamente 'museos'")
        print("3. No se pierde información importante cuando el usuario indica que no quiere dar más detalles")
    else:
        print("\n⚠️ Algunas pruebas fallaron. Revisar los logs anteriores.")