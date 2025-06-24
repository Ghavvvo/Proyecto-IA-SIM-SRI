"""
Sistema simplificado de experimentación para comparar las 3 versiones del agente
Enfocado en métricas fundamentales: media, desviación estándar, distribuciones y tests estadísticos
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

# Añadir el directorio padre al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))




# Configuración de matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# Configuración del experimento
NUM_REPLICAS = 50  # Número de réplicas por escenario
SEED_BASE = 42
ALPHA = 0.05  # Nivel de significancia para tests estadísticos

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
    
    # Inicializar generador de datos
    generator = TestDataGenerator(seed=SEED_BASE)
    
    # Almacenar resultados
    resultados = {
        'v1': {'satisfaccion': [], 'cansancio': [], 'tiempo': [], 'tipos_ruta': []},
        'v2': {'satisfaccion': [], 'cansancio': [], 'tiempo': [], 'tipos_ruta': []},
        'v3': {'satisfaccion': [], 'cansancio': [], 'tiempo': [], 'tipos_ruta': []}
    }
    
    # Generar conjunto de escenarios variados usando distribución normal
    print("\nGenerando escenarios de prueba con distribución normal...")
    escenarios = []
    
    # Definir tipos de escenarios y sus funciones generadoras
    tipos_escenarios = [
        ('corto', generator.generar_itinerario_corto),
        ('medio', generator.generar_itinerario_medio),
        ('largo', generator.generar_itinerario_largo),
        ('extremo_optimo', lambda: generator.generar_escenario_extremo("optimo")),
        ('extremo_pesimo', lambda: generator.generar_escenario_extremo("pesimo"))
    ]
    
    # Parámetros de la distribución normal para selección de tipo de escenario
    # Variable aleatoria X ~ N(μ=2.0, σ=1.2)
    # donde X representa el "índice de complejidad del escenario"
    media_tipo_escenario = 2.0  # μ: centrado en escenarios medios
    desviacion_tipo_escenario = 1.2  # σ: permite variación pero favorece el centro
    
    # Contador de tipos para estadísticas
    contador_tipos = {tipo: 0 for tipo, _ in tipos_escenarios}
    
    print(f"\nVariable aleatoria para selección: X ~ N(μ={media_tipo_escenario}, σ={desviacion_tipo_escenario})")
    print("Mapeo de valores:")
    print("  X < 0.5: Escenario corto")
    print("  0.5 ≤ X < 1.5: Escenario medio")
    print("  1.5 ≤ X < 2.5: Escenario medio") 
    print("  2.5 ≤ X < 3.5: Escenario largo")
    print("  X ≥ 3.5: Escenario extremo")
    
    for i in range(NUM_REPLICAS):
        # Generar valor de la variable aleatoria X ~ N(μ=2.0, σ=1.2)
        valor_x = np.random.normal(media_tipo_escenario, desviacion_tipo_escenario)
        
        # Convertir el valor continuo X a índice discreto (0-4) usando límites
        if valor_x < 0.5:
            indice = 0  # corto
        elif valor_x < 1.5:
            indice = 1  # medio (más probable)
        elif valor_x < 2.5:
            indice = 2  # medio (más probable)
        elif valor_x < 3.5:
            indice = 3  # largo
        else:
            # Valores extremos: decidir entre óptimo o pésimo
            indice = 4 if valor_x > media_tipo_escenario else 3
        
        # Asegurar que el índice esté en rango válido
        indice = max(0, min(len(tipos_escenarios) - 1, indice))
        
        # Generar escenario del tipo seleccionado
        tipo_nombre, generador_func = tipos_escenarios[indice]
        escenario = generador_func()
        contador_tipos[tipo_nombre] += 1
        
        # Agregar tipo de ruta al escenario
        escenario['tipo_ruta'] = tipo_nombre
        
        # Ocasionalmente (20% de probabilidad) asignar un perfil de turista específico
        if np.random.random() < 0.2:
            perfil = np.random.choice(["exigente", "relajado", "average"])
            escenario["perfil_turista_forzado"] = perfil
            escenario["nombre_escenario"] += f" - Turista {perfil}"
        
        escenarios.append(escenario)
    
    # Mostrar distribución de tipos generados
    print("\nDistribución de escenarios generados:")
    for tipo, count in contador_tipos.items():
        porcentaje = (count / NUM_REPLICAS) * 100
        print(f"  - {tipo}: {count} ({porcentaje:.1f}%)")
    
    # Ejecutar simulaciones
    print(f"\nEjecutando simulaciones con {len(escenarios)} escenarios variados...")
    
    for replica, escenario in enumerate(escenarios):
        # Establecer semilla para reproducibilidad parcial
        random.seed(SEED_BASE + replica)
        np.random.seed(SEED_BASE + replica)
        
        # Extraer itinerario y contexto del escenario
        itinerario = escenario['itinerario']
        contexto = escenario['contexto']
        
        # Simular con cada versión
        for version, AgentClass in [
            ('v1', TouristSimulationAgentV1),
            ('v2', TouristSimulationAgentV2),
            ('v3', TouristSimulationAgentV3)
        ]:
            # Crear agente con perfil específico si está definido
            perfil = escenario.get('perfil_turista_forzado', 'average')
            agente = AgentClass(f"agent_{version}", perfil)
            
            metricas = ejecutar_simulacion(agente, itinerario, contexto)
            
            resultados[version]['satisfaccion'].append(metricas['satisfaccion_general'])
            resultados[version]['cansancio'].append(metricas['cansancio_final'])
            resultados[version]['tiempo'].append(metricas['tiempo_ejecucion'])
            resultados[version]['tipos_ruta'].append(escenario['tipo_ruta'])
        
        if (replica + 1) % 10 == 0:
            print(f"  Completadas {replica + 1}/{NUM_REPLICAS} réplicas")
            print(f"    Último escenario: {escenario['nombre_escenario']}")
    
    print("\nSimulaciones completadas!")
    print(f"Total de escenarios únicos utilizados: {len(set(e['nombre_escenario'] for e in escenarios))}")
    
    # Realizar análisis estadístico completo
    analizar_resultados_estadisticos(resultados, contador_tipos)
    
    return resultados

def calcular_estadisticas_descriptivas(datos):
    """Calcula medidas de tendencia central y dispersión"""
    datos_array = np.array(datos)
    
    # Medidas de tendencia central
    media = np.mean(datos_array)
    mediana = np.median(datos_array)
    
    # Calcular moda (valor más frecuente)
    try:
        moda_resultado = stats.mode(datos_array, keepdims=True)
        moda = moda_resultado.mode[0]
        frecuencia_moda = moda_resultado.count[0]
    except:
        # Si no hay moda clara, usar el valor más cercano a la media
        moda = media
        frecuencia_moda = 1
    
    # Medidas de dispersión
    desviacion_std = np.std(datos_array, ddof=1)  # Desviación estándar muestral
    varianza = np.var(datos_array, ddof=1)
    rango = np.max(datos_array) - np.min(datos_array)
    
    return {
        'media': media,
        'mediana': mediana,
        'moda': moda,
        'frecuencia_moda': frecuencia_moda,
        'desviacion_std': desviacion_std,
        'varianza': varianza,
        'rango': rango,
        'min': np.min(datos_array),
        'max': np.max(datos_array),
        'n': len(datos_array)
    }

def test_normalidad(datos, nombre_metrica, version):
    """Realiza test de normalidad Shapiro-Wilk"""
    datos_array = np.array(datos)
    
    if len(datos_array) < 3:
        return None, None, "Insuficientes datos para test"
    
    # Test de Shapiro-Wilk
    estadistico, p_valor = stats.shapiro(datos_array)
    
    # Interpretación
    es_normal = p_valor > ALPHA
    interpretacion = f"{'Normal' if es_normal else 'No normal'} (α={ALPHA})"
    
    return estadistico, p_valor, interpretacion

def generar_grafico_distribucion_rutas(contador_tipos):
    """Genera gráfico de distribución de tipos de ruta"""
    plt.figure(figsize=(12, 8))
    
    # Preparar datos
    tipos = list(contador_tipos.keys())
    cantidades = list(contador_tipos.values())
    colores = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    # Crear gráfico de barras
    plt.subplot(2, 2, 1)
    bars = plt.bar(tipos, cantidades, color=colores[:len(tipos)])
    plt.title('Distribución de Tipos de Ruta Generados', fontsize=14, fontweight='bold')
    plt.xlabel('Tipo de Ruta')
    plt.ylabel('Cantidad')
    plt.xticks(rotation=45)
    
    # Agregar valores en las barras
    for bar, cantidad in zip(bars, cantidades):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{cantidad}', ha='center', va='bottom', fontweight='bold')
    
    # Crear gráfico de pastel
    plt.subplot(2, 2, 2)
    plt.pie(cantidades, labels=tipos, colors=colores[:len(tipos)], autopct='%1.1f%%',
            startangle=90)
    plt.title('Proporción de Tipos de Ruta', fontsize=14, fontweight='bold')
    
    # Estadísticas adicionales
    plt.subplot(2, 1, 2)
    plt.axis('off')
    
    # Crear tabla de estadísticas
    total = sum(cantidades)
    stats_text = "ESTADÍSTICAS DE DISTRIBUCIÓN DE RUTAS\n" + "="*50 + "\n"
    stats_text += f"Total de escenarios: {total}\n\n"
    
    for tipo, cantidad in zip(tipos, cantidades):
        porcentaje = (cantidad / total) * 100
        stats_text += f"{tipo.upper():<20}: {cantidad:>3} ({porcentaje:>5.1f}%)\n"
    
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
             fontfamily='monospace', fontsize=11, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('distribucion_tipos_ruta.png', dpi=300, bbox_inches='tight')
    plt.show()

def generar_boxplots_comparativos(resultados):
    """Genera boxplots para comparar distribuciones entre versiones"""
    metricas = ['satisfaccion', 'cansancio', 'tiempo']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, metrica in enumerate(metricas):
        # Preparar datos para boxplot
        datos_boxplot = []
        etiquetas = []
        
        for version in ['v1', 'v2', 'v3']:
            datos_boxplot.append(resultados[version][metrica])
            etiquetas.append(f'V{version[-1]}')
        
        # Crear boxplot
        bp = axes[i].boxplot(datos_boxplot, labels=etiquetas, patch_artist=True)
        
        # Personalizar colores
        colores = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        for patch, color in zip(bp['boxes'], colores):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[i].set_title(f'Distribución de {metrica.title()}', fontsize=14, fontweight='bold')
        axes[i].set_ylabel(metrica.title())
        axes[i].grid(True, alpha=0.3)
        
        # Agregar estadísticas básicas
        for j, version in enumerate(['v1', 'v2', 'v3']):
            datos = resultados[version][metrica]
            media = np.mean(datos)
            axes[i].text(j+1, media, f'μ={media:.2f}', 
                        ha='center', va='bottom', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('boxplots_comparativos.png', dpi=300, bbox_inches='tight')
    plt.show()

def generar_histogramas_normalidad(resultados):
    """Genera histogramas para verificar normalidad de datos"""
    metricas = ['satisfaccion', 'cansancio', 'tiempo']
    versiones = ['v1', 'v2', 'v3']
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    for i, metrica in enumerate(metricas):
        for j, version in enumerate(versiones):
            datos = np.array(resultados[version][metrica])
            
            # Crear histograma
            axes[i, j].hist(datos, bins=10, density=True, alpha=0.7, 
                           color=['#FF6B6B', '#4ECDC4', '#45B7D1'][j])
            
            # Superponer curva normal teórica
            mu, sigma = np.mean(datos), np.std(datos)
            x = np.linspace(datos.min(), datos.max(), 100)
            y = stats.norm.pdf(x, mu, sigma)
            axes[i, j].plot(x, y, 'r-', linewidth=2, label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')
            
            # Test de normalidad
            stat, p_val, interpretacion = test_normalidad(datos, metrica, version)
            
            axes[i, j].set_title(f'{metrica.title()} - V{version[-1]}\n{interpretacion}', 
                               fontsize=12, fontweight='bold')
            axes[i, j].set_xlabel(metrica.title())
            axes[i, j].set_ylabel('Densidad')
            axes[i, j].legend()
            axes[i, j].grid(True, alpha=0.3)
            
            # Agregar estadísticas en el gráfico
            stats_text = f'n={len(datos)}\nW={stat:.4f}\np={p_val:.4f}' if stat else 'N/A'
            axes[i, j].text(0.02, 0.98, stats_text, transform=axes[i, j].transAxes,
                           verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('histogramas_normalidad.png', dpi=300, bbox_inches='tight')
    plt.show()

def analizar_resultados_estadisticos(resultados, contador_tipos):
    """Realiza análisis estadístico completo de los resultados"""
    print("\n" + "="*80)
    print("ANÁLISIS ESTADÍSTICO COMPLETO")
    print("="*80)
    
    # 1. Gráfico de distribución de tipos de ruta
    print("\n1. GENERANDO GRÁFICO DE DISTRIBUCIÓN DE TIPOS DE RUTA...")
    generar_grafico_distribucion_rutas(contador_tipos)
    
    # 2. Estadísticas descriptivas por métrica y versión
    print("\n2. ESTADÍSTICAS DESCRIPTIVAS POR MÉTRICA Y VERSIÓN")
    print("-" * 60)
    
    metricas = ['satisfaccion', 'cansancio', 'tiempo']
    versiones = ['v1', 'v2', 'v3']
    
    for metrica in metricas:
        print(f"\n📊 MÉTRICA: {metrica.upper()}")
        print("=" * 50)
        
        for version in versiones:
            datos = resultados[version][metrica]
            stats_desc = calcular_estadisticas_descriptivas(datos)
            
            print(f"\n🔹 Versión {version.upper()}:")
            print(f"   Tendencia Central:")
            print(f"     • Media:    {stats_desc['media']:.4f}")
            print(f"     • Mediana:  {stats_desc['mediana']:.4f}")
            print(f"     • Moda:     {stats_desc['moda']:.4f} (freq: {stats_desc['frecuencia_moda']})")
            print(f"   Dispersión:")
            print(f"     • Desv. Std: {stats_desc['desviacion_std']:.4f}")
            print(f"     • Varianza:  {stats_desc['varianza']:.4f}")
            print(f"     • Rango:     {stats_desc['rango']:.4f} [{stats_desc['min']:.2f}, {stats_desc['max']:.2f}]")
            print(f"   Tamaño muestral: n = {stats_desc['n']}")
    
    # 3. Verificación de normalidad
    print("\n3. VERIFICACIÓN DE SUPUESTOS - TEST DE NORMALIDAD")
    print("-" * 60)
    print("Test de Shapiro-Wilk (H₀: Los datos siguen distribución normal)")
    
    for metrica in metricas:
        print(f"\n📈 {metrica.upper()}:")
        for version in versiones:
            datos = resultados[version][metrica]
            stat, p_val, interpretacion = test_normalidad(datos, metrica, version)
            
            if stat:
                print(f"   V{version[-1]}: W = {stat:.4f}, p = {p_val:.4f} → {interpretacion}")
            else:
                print(f"   V{version[-1]}: {interpretacion}")
    
    # 4. Generar visualizaciones
    print("\n4. GENERANDO VISUALIZACIONES CLAVE...")
    print("   • Boxplots comparativos...")
    generar_boxplots_comparativos(resultados)
    
    print("   • Histogramas para verificación de normalidad...")
    generar_histogramas_normalidad(resultados)
    
    # 5. Resumen de recomendaciones
    print("\n5. RECOMENDACIONES PARA ANÁLISIS POSTERIOR")
    print("-" * 60)
    print("📋 Basado en los tests de normalidad:")
    
    for metrica in metricas:
        print(f"\n   {metrica.upper()}:")
        tests_normales = 0
        for version in versiones:
            datos = resultados[version][metrica]
            _, p_val, _ = test_normalidad(datos, metrica, version)
            if p_val and p_val > ALPHA:
                tests_normales += 1
        
        if tests_normales == 3:
            print("     ✅ Todas las versiones muestran normalidad")
            print("     → Usar tests paramétricos (t-test, ANOVA)")
        elif tests_normales >= 1:
            print("     ⚠️  Normalidad mixta entre versiones")
            print("     → Considerar tests no paramétricos o transformaciones")
        else:
            print("     ❌ Ninguna versión muestra normalidad")
            print("     → Usar tests no paramétricos (Mann-Whitney, Kruskal-Wallis)")
    
    print(f"\n📁 Gráficos guardados:")
    print("   • distribucion_tipos_ruta.png")
    print("   • boxplots_comparativos.png") 
    print("   • histogramas_normalidad.png")
    
    print("\n" + "="*80)
    print("ANÁLISIS ESTADÍSTICO COMPLETADO")
    print("="*80)

if __name__ == "__main__":
    # Ejecutar experimento
    resultados = ejecutar_experimento_completo()
    
    print("\n🎯 Experimento finalizado exitosamente!")
    print("📊 Revisa los gráficos generados para análisis detallado.")

