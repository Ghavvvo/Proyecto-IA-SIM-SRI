"""
Sistema de experimentación simplificado para generar una única diapositiva con análisis completo
Incluye: distribución de rutas, medias de satisfacción, varianza con boxplot, y normalidad
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import random
import time
from datetime import datetime
from agents.agent_simulation.agent_simulation_v1 import TouristSimulationAgentV1
from agents.agent_simulation.agent_simulation_v2 import TouristSimulationAgentV2
from agents.agent_simulation.agent_simulation import TouristSimulationAgent as TouristSimulationAgentV3
from tests.test_data_generator import TestDataGenerator


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10


NUM_REPLICAS = 50  
SEED_BASE = 42
ALPHA = 0.05  

def ejecutar_simulacion(agente, itinerario, contexto):
    """Ejecuta una simulación y retorna métricas clave"""
    inicio = time.time()
    resultado = agente.simular_itinerario(itinerario, contexto)
    tiempo_ejecucion = time.time() - inicio
    
    return {
        'satisfaccion_general': resultado['satisfaccion_general'],
        'cansancio_final': resultado['cansancio_final'],
        'tiempo_ejecucion': tiempo_ejecucion,
        'num_lugares': len(resultado['lugares_visitados']),
        'satisfacciones_lugares': [l['satisfaccion'] for l in resultado['lugares_visitados']]
    }

def ejecutar_experimento_completo():
    """Ejecuta el experimento completo con las 3 versiones"""
    print("=" * 80)
    print("EXPERIMENTO DE COMPARACIÓN DE AGENTES DE SIMULACIÓN")
    print("=" * 80)
    print(f"\nConfiguración: {NUM_REPLICAS} réplicas por versión")
    print("Versiones a comparar: V1 (básica), V2 (intermedia), V3 (avanzada)")
    
    
    generator = TestDataGenerator(seed=SEED_BASE)
    
    
    resultados = {
        'v1': {'satisfaccion': [], 'tipos_ruta': []},
        'v2': {'satisfaccion': [], 'tipos_ruta': []},
        'v3': {'satisfaccion': [], 'tipos_ruta': []}
    }
    
    
    print("\nGenerando escenarios de prueba con distribución normal...")
    escenarios = []
    
    
    tipos_escenarios = [
        ('corto', generator.generar_itinerario_corto),
        ('medio', generator.generar_itinerario_medio),
        ('largo', generator.generar_itinerario_largo),
        ('extremo_optimo', lambda: generator.generar_escenario_extremo("optimo")),
        ('extremo_pesimo', lambda: generator.generar_escenario_extremo("pesimo"))
    ]
    
    
    media_tipo_escenario = 2.0
    desviacion_tipo_escenario = 1.2
    
    
    contador_tipos = {tipo: 0 for tipo, _ in tipos_escenarios}
    
    for i in range(NUM_REPLICAS):
        
        valor_x = np.random.normal(media_tipo_escenario, desviacion_tipo_escenario)
        
        
        if valor_x < 0.5:
            indice = 0  
        elif valor_x < 1.5:
            indice = 1  
        elif valor_x < 2.5:
            indice = 2  
        elif valor_x < 3.5:
            indice = 3  
        else:
            indice = 4  
        
        
        indice = max(0, min(len(tipos_escenarios) - 1, indice))
        
        
        tipo_nombre, generador_func = tipos_escenarios[indice]
        escenario = generador_func()
        contador_tipos[tipo_nombre] += 1
        
        
        escenario['tipo_ruta'] = tipo_nombre
        
        
        if np.random.random() < 0.2:
            perfil = np.random.choice(["exigente", "relajado", "average"])
            escenario["perfil_turista_forzado"] = perfil
            escenario["nombre_escenario"] += f" - Turista {perfil}"
        
        escenarios.append(escenario)
    
    
    print(f"\nEjecutando simulaciones con {len(escenarios)} escenarios variados...")
    
    for replica, escenario in enumerate(escenarios):
        
        random.seed(SEED_BASE + replica)
        np.random.seed(SEED_BASE + replica)
        
        
        itinerario = escenario['itinerario']
        contexto = escenario['contexto']
        
        
        for version, AgentClass in [
            ('v1', TouristSimulationAgentV1),
            ('v2', TouristSimulationAgentV2),
            ('v3', TouristSimulationAgentV3)
        ]:
            
            perfil = escenario.get('perfil_turista_forzado', 'average')
            agente = AgentClass(f"agent_{version}", perfil)
            
            metricas = ejecutar_simulacion(agente, itinerario, contexto)
            
            resultados[version]['satisfaccion'].append(metricas['satisfaccion_general'])
            resultados[version]['tipos_ruta'].append(escenario['tipo_ruta'])
        
        if (replica + 1) % 10 == 0:
            print(f"  Completadas {replica + 1}/{NUM_REPLICAS} réplicas")
    
    print("\nSimulaciones completadas!")
    
    
    generar_diapositiva_completa(resultados, contador_tipos)
    
    return resultados

def test_normalidad(datos):
    """Realiza test de normalidad Shapiro-Wilk"""
    datos_array = np.array(datos)
    
    if len(datos_array) < 3:
        return None, None, False
    
    
    estadistico, p_valor = stats.shapiro(datos_array)
    
    
    es_normal = p_valor > ALPHA
    
    return estadistico, p_valor, es_normal

def generar_diapositiva_completa(resultados, contador_tipos):
    """
    Genera una única diapositiva con los 4 elementos solicitados:
    1. Distribución de rutas usadas
    2. Gráfico de medias de satisfacción
    3. Varianza de satisfacción con boxplot
    4. Distribución y normalidad de satisfacción por versión
    """
    
    
    fig = plt.figure(figsize=(20, 16))
    
    
    colores_versiones = ['
    versiones = ['v1', 'v2', 'v3']
    etiquetas_versiones = ['V1 (Básica)', 'V2 (Intermedia)', 'V3 (Avanzada)']
    
    
    ax1 = plt.subplot(2, 2, 1)
    tipos = list(contador_tipos.keys())
    cantidades = list(contador_tipos.values())
    colores_rutas = ['
    
    
    bars = ax1.bar(tipos, cantidades, color=colores_rutas[:len(tipos)], alpha=0.8, edgecolor='black')
    ax1.set_title('Distribución de Tipos de Ruta Utilizados', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Tipo de Ruta', fontsize=12)
    ax1.set_ylabel('Cantidad de Escenarios', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    
    for bar, cantidad in zip(bars, cantidades):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{cantidad}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    
    total_escenarios = sum(cantidades)
    for i, (bar, cantidad) in enumerate(zip(bars, cantidades)):
        porcentaje = (cantidad / total_escenarios) * 100
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f'{porcentaje:.1f}%', ha='center', va='center', 
                fontweight='bold', fontsize=9, color='white')
    
    ax1.grid(True, alpha=0.3, axis='y')
    
    
    ax2 = plt.subplot(2, 2, 2)
    medias = []
    errores_std = []
    
    for version in versiones:
        datos_satisfaccion = resultados[version]['satisfaccion']
        media = np.mean(datos_satisfaccion)
        std = np.std(datos_satisfaccion, ddof=1)
        medias.append(media)
        errores_std.append(std)
    
    
    bars = ax2.bar(etiquetas_versiones, medias, color=colores_versiones, alpha=0.8, 
                   yerr=errores_std, capsize=5, edgecolor='black')
    
    ax2.set_title('Medias de Satisfacción por Versión del Agente', fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylabel('Satisfacción Media (0-10)', fontsize=12)
    ax2.set_ylim(0, 10)
    ax2.tick_params(axis='x', rotation=15)
    
    
    for bar, media, std in zip(bars, medias, errores_std):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.1,
                f'{media:.2f}±{std:.2f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
    
    ax2.grid(True, alpha=0.3, axis='y')
    
    
    ax3 = plt.subplot(2, 2, 3)
    datos_boxplot = [resultados[version]['satisfaccion'] for version in versiones]
    
    
    bp = ax3.boxplot(datos_boxplot, labels=etiquetas_versiones, patch_artist=True, 
                     showmeans=True, meanline=True)
    
    
    for patch, color in zip(bp['boxes'], colores_versiones):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    
    for element in ['whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(bp[element], color='black', linewidth=1.5)
    
    plt.setp(bp['means'], color='red', linewidth=2)
    
    ax3.set_title('Varianza de Satisfacción - Análisis de Dispersión', fontsize=14, fontweight='bold', pad=20)
    ax3.set_ylabel('Satisfacción (0-10)', fontsize=12)
    ax3.tick_params(axis='x', rotation=15)
    ax3.grid(True, alpha=0.3, axis='y')
    
    
    for i, version in enumerate(versiones):
        datos = resultados[version]['satisfaccion']
        varianza = np.var(datos, ddof=1)
        ax3.text(i+1, 1, f'σ²={varianza:.3f}', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                fontsize=9, fontweight='bold')
    
    
    ax4 = plt.subplot(2, 2, 4)
    
    
    for i, version in enumerate(versiones):
        datos = np.array(resultados[version]['satisfaccion'])
        
        
        ax4.hist(datos, bins=12, alpha=0.6, color=colores_versiones[i], 
                label=etiquetas_versiones[i], density=True, edgecolor='black')
        
        
        mu, sigma = np.mean(datos), np.std(datos)
        x = np.linspace(datos.min(), datos.max(), 100)
        y = stats.norm.pdf(x, mu, sigma)
        ax4.plot(x, y, '--', color=colores_versiones[i], linewidth=2, alpha=0.8)
    
    ax4.set_title('Distribución de Satisfacción y Test de Normalidad', fontsize=14, fontweight='bold', pad=20)
    ax4.set_xlabel('Satisfacción (0-10)', fontsize=12)
    ax4.set_ylabel('Densidad', fontsize=12)
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    
    normalidad_text = ""
    for i, version in enumerate(versiones):
        datos = resultados[version]['satisfaccion']
        stat, p_val, es_normal = test_normalidad(datos)
        
        if stat is not None:
            resultado = "Normal" if es_normal else "No Normal"
            normalidad_text += f"V{i+1}: {resultado}\n(p={p_val:.3f})\n\n"
        else:
            normalidad_text += f"V{i+1}: N/A\n\n"
    
    
    ax4.text(0.98, 0.98, normalidad_text.strip(), transform=ax4.transAxes,
             verticalalignment='top', horizontalalignment='right', 
             fontfamily='monospace', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.95, edgecolor='black'))
    
    
    plt.tight_layout(pad=3.0)

    
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'analisis_completo_agentes_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"\n📊 Diapositiva generada exitosamente: {filename}")
    
    
    print("\n" + "="*80)
    print("RESUMEN ESTADÍSTICO")
    print("="*80)
    
    print(f"\n📈 DISTRIBUCIÓN DE RUTAS:")
    for tipo, cantidad in contador_tipos.items():
        porcentaje = (cantidad / sum(contador_tipos.values())) * 100
        print(f"   {tipo.upper():<15}: {cantidad:>3} ({porcentaje:>5.1f}%)")
    
    print(f"\n📊 SATISFACCIÓN POR VERSIÓN:")
    for i, version in enumerate(versiones):
        datos = resultados[version]['satisfaccion']
        media = np.mean(datos)
        std = np.std(datos, ddof=1)
        varianza = np.var(datos, ddof=1)
        
        
        stat, p_val, es_normal = test_normalidad(datos)
        normalidad_str = "Normal" if es_normal else "No Normal" if stat else "N/A"
        
        print(f"   {etiquetas_versiones[i]:<25}:")
        print(f"     • Media: {media:.3f} ± {std:.3f}")
        print(f"     • Varianza: {varianza:.3f}")
        print(f"     • Normalidad: {normalidad_str}")
        if stat:
            print(f"     • Shapiro-Wilk: W={stat:.4f}, p={p_val:.4f}")
    
    plt.show()
    
    return filename

if __name__ == "__main__":
    
    print("Iniciando experimento de comparación de agentes...")
    resultados = ejecutar_experimento_completo()
    
    print("\n🎯 Experimento finalizado exitosamente!")
    print("📊 Revisa la diapositiva generada para el análisis completo.")