"""
Test simple para verificar el cambio en el flujo de preferencias
"""

import re

def check_preferences_flow():
    """
    Verifica que el código modificado tenga el flujo correcto
    """
    print("🔍 Verificando cambios en agent_coordinator.py...")
    
    
    with open('agent_coordinator.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    
    method_pattern = r'def _execute_aco_search_with_preferences\(self, preferences: dict\) -> str:(.*?)(?=\n    def|\Z)'
    match = re.search(method_pattern, content, re.DOTALL)
    
    if not match:
        print("❌ No se encontró el método _execute_aco_search_with_preferences")
        return False
    
    method_content = match.group(1)
    
    
    checks = {
        "PASO 1 - Consulta BD local": "Consultando información existente en la base de datos" in method_content,
        "Consulta al RAG primero": "self.rag_agent.receive" in method_content and method_content.index("self.rag_agent.receive") < method_content.index("search_google_aco") if "search_google_aco" in method_content else True,
        "Evaluación de utilidad": "_evaluate_response_usefulness" in method_content,
        "Mensaje BD suficiente": "Encontré suficiente información en la base de datos local" in method_content,
        "Mensaje BD insuficiente": "La información en la base de datos no es suficiente" in method_content,
        "Búsqueda condicional": "if evaluation:" in method_content,
    }
    
    print("\n📋 Resultados de la verificación:")
    all_passed = True
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✅ ÉXITO: El flujo ha sido modificado correctamente")
        print("\n📝 Comportamiento esperado:")
        print("  1. Al obtener preferencias, primero consulta la BD local")
        print("  2. Evalúa si la información es suficiente")
        print("  3. Solo busca en DuckDuckGo si la información no es suficiente")
    else:
        print("\n❌ ERROR: Algunos cambios no se aplicaron correctamente")
    
    
    print("\n📄 Fragmento del código modificado:")
    lines = method_content.split('\n')[:20]  
    for i, line in enumerate(lines):
        if "Consultando información existente" in line or "_evaluate_response_usefulness" in line:
            print(f"  >>> {line.strip()}")
        else:
            print(f"      {line.strip()}")

if __name__ == "__main__":
    check_preferences_flow()