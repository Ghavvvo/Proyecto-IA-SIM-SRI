#!/usr/bin/env python3
"""
Script para solucionar el warning de flaml.automl
"""

import subprocess
import sys

def main():
    print("=== Solución para el warning de flaml.automl ===\n")
    print("El warning aparece porque autogen-agentchat importa flaml,")
    print("pero flaml está instalado sin las funcionalidades de AutoML.\n")
    
    print("Opciones disponibles:")
    print("1. Instalar flaml con automl extras (si necesitas AutoML)")
    print("2. Suprimir el warning (si NO necesitas AutoML)")
    print("3. Cancelar\n")
    
    choice = input("Seleccione una opción (1/2/3): ")
    
    if choice == '1':
        print("\nInstalando flaml con automl extras...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'flaml[automl]'])
            print("\n✓ flaml[automl] instalado exitosamente")
            print("El warning no debería aparecer más.")
        except subprocess.CalledProcessError:
            print("\n✗ Error al instalar flaml[automl]")
    
    elif choice == '2':
        print("\nPara suprimir el warning, agregue este código al inicio de su script principal:\n")
        print("import warnings")
        print("warnings.filterwarnings('ignore', message='flaml.automl is not available')")
        print("\nO puede agregar esta variable de entorno antes de ejecutar:")
        print("export PYTHONWARNINGS='ignore::UserWarning:flaml'")
        
        add_to_main = input("\n¿Desea agregar el código de supresión a main.py? (s/n): ")
        if add_to_main.lower() == 's':
            add_warning_suppression()
    
    else:
        print("\nOperación cancelada.")

def add_warning_suppression():
    """Agrega código para suprimir el warning en main.py"""
    try:
        with open('main.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Buscar dónde insertar el código
        lines = content.split('\n')
        insert_index = 0
        
        # Buscar después de los imports existentes
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith('#') and not line.startswith('import') and not line.startswith('from'):
                insert_index = i
                break
        
        # Preparar el código a insertar
        warning_code = [
            "",
            "# Suprimir warning de flaml.automl",
            "import warnings",
            "warnings.filterwarnings('ignore', message='flaml.automl is not available')",
            ""
        ]
        
        # Insertar el código
        for j, code_line in enumerate(warning_code):
            lines.insert(insert_index + j, code_line)
        
        # Escribir el archivo actualizado
        with open('main.py', 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print("\n✓ Código de supresión agregado a main.py")
        
    except Exception as e:
        print(f"\n✗ Error al modificar main.py: {e}")

if __name__ == "__main__":
    main()