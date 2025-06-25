import re
import json
import statistics
import math
from typing import Dict, Any, List
from core.mistral_config import MistralClient
import json

def format_as_simulation_input(raw_response: str, preferences: dict) -> Dict[str, Any]:
    """
    Uses Mistral to transform a formatted itinerary into a structured format optimal for tourist simulation.
    Enriched with detailed information for simulation including types, popularity, distances, and timing.
    Enhanced to capture travel times and activity durations from the itinerary format.
    """
    try:
        mistral_client = MistralClient(model_name="flash")
        prompt = f"""
        Eres un experto en simulación de turistas. Recibe el siguiente itinerario de viaje y extrae información detallada para simular a un turista siguiendo el itinerario.

        Itinerario:
        {raw_response}

        Preferencias del viajero:
        {preferences}

        INSTRUCCIONES ESPECÍFICAS PARA CAPTURAR TIEMPOS:
        - Busca horarios específicos en formato 🕐 9:00, 🕑 12:30, etc.
        - Identifica duraciones de actividades en paréntesis como (1.5 horas), (30 min), (1 hora)
        - Detecta tiempos de desplazamiento como "🚗 **Desplazamiento (3 horas en auto o bus...)**"
        - Convierte todas las duraciones a horas decimales: 30 min = 0.5, 1.5 horas = 1.5

        INSTRUCCIONES:
        - Devuelve un JSON estructurado con la siguiente forma:
          {{
            "destination": string,
            "season": string (verano/invierno/primavera/otoño, inferir de las fechas o usar "verano" por defecto),
            "tourist_profile": string (exigente/relajado/average basado en preferencias),
            "days": [
              {{
                "day": int (número del día, empezando desde 1),
                "title": string (titulo del dia),
                "day_of_week": string (lunes/martes/etc, usar "sabado" si no se especifica),
                "activities": [
                  {{
                    "time": string (formato HH:MM, extraer del emoji de reloj o usar secuencial),
                    "location": string (nombre del lugar o actividad),
                    "description": string (descripcion breve del lugar o actividad),
                    "type": string (museo/restaurante/parque/monumento/playa/hotel/centro_comercial/teatro/zoo/desplazamiento/otro),
                    "popularity": float (1-10, estimar basado en la descripción, lugares famosos 8-10, normales 5-7),
                    "estimated_duration_hours": float (EXTRAER del texto en paréntesis, convertir a horas decimales),
                    "distance_from_previous_km": float (para desplazamientos, extraer distancia o tiempo de viaje),
                    "travel_time_hours": float (SOLO para actividades de desplazamiento, extraer tiempo de viaje),
                    "is_travel": boolean (true si es una actividad de desplazamiento, false si es visita a lugar)
                  }}
                ]
              }}
            ],
            "interests": list (extraer de preferencias),
            "budget": string (si está disponible),
            "duration": string (duración total del viaje)
          }}
        
        REGLAS ADICIONALES:
        - Para actividades de desplazamiento: type="desplazamiento", is_travel=true, travel_time_hours=tiempo extraído
        - Para actividades normales: is_travel=false, estimated_duration_hours=tiempo extraído del paréntesis
        - Extraer horarios exactos de los emojis de reloj (🕐 9:00 = "09:00")
        - Convertir duraciones: "30 min" = 0.5, "1 hora" = 1.0, "1.5 horas" = 1.5, "3 horas" = 3.0
        - Para tourist_profile: si menciona lujo/exclusivo = "exigente", si menciona relajado/tranquilo = "relajado", sino "average"
        - Para popularidad: lugares muy conocidos o mencionados como "imperdibles" = 8-10
        - Para distancias de desplazamiento: extraer del texto o estimar según el tiempo de viaje
        - Devuelve SOLO el JSON, sin explicaciones ni comentarios.
        """
        response = mistral_client.generate(prompt)
        json_str = response.strip()
        
        start_idx = json_str.find('{')
        end_idx = json_str.rfind('}') + 1
        if start_idx != -1 and end_idx > start_idx:
            json_str = json_str[start_idx:end_idx]

            simulation_data = json.loads(json_str)
            
            
            if 'season' not in simulation_data:
                simulation_data['season'] = 'verano'
            if 'tourist_profile' not in simulation_data:
                simulation_data['tourist_profile'] = 'average'
                
            
            for day in simulation_data.get('days', []):
                if 'day_of_week' not in day:
                    day['day_of_week'] = 'sabado'
                    
                for i, activity in enumerate(day.get('activities', [])):
                    if 'type' not in activity:
                        activity['type'] = _infer_activity_type(activity.get('location', ''))
                    if 'popularity' not in activity:
                        activity['popularity'] = 7.0
                    if 'estimated_duration_hours' not in activity:
                        activity['estimated_duration_hours'] = _get_default_duration(activity.get('type', 'otro'))
                    if 'distance_from_previous_km' not in activity:
                        activity['distance_from_previous_km'] = 3.0 if i == 0 else 2.0
                    if 'time' not in activity or not activity['time']:
                        
                        base_hours = [9, 11, 14, 16, 18, 20]
                        activity['time'] = f"{base_hours[i % len(base_hours)]:02d}:00"
                    
                    
                    if 'is_travel' not in activity:
                        activity['is_travel'] = activity.get('type', 'otro') == 'desplazamiento'
                    if 'travel_time_hours' not in activity:
                        activity['travel_time_hours'] = activity.get('estimated_duration_hours', 0.0) if activity.get('is_travel', False) else 0.0
            
            return simulation_data
        return {}
    except Exception as e:
        print(f"Error usando Mistral para estructurar simulación: {e}")
        return {}

def _infer_activity_type(location_name: str) -> str:
    """Infiere el tipo de actividad basado en el nombre del lugar"""
    location_lower = location_name.lower()
    
    type_keywords = {
        'desplazamiento': ['desplazamiento', 'viaje', 'traslado', 'transporte', 'auto', 'bus', 'avión', 'tren'],
        'museo': ['museo', 'galería', 'exposición'],
        'restaurante': ['restaurante', 'café', 'cafetería', 'comida', 'almuerzo', 'cena'],
        'parque': ['parque', 'jardín', 'plaza'],
        'monumento': ['monumento', 'estatua', 'memorial', 'catedral', 'iglesia'],
        'playa': ['playa', 'costa', 'mar'],
        'hotel': ['hotel', 'hospedaje', 'alojamiento'],
        'centro_comercial': ['centro comercial', 'shopping', 'tiendas'],
        'teatro': ['teatro', 'cine', 'espectáculo'],
        'zoo': ['zoológico', 'zoo', 'acuario']
    }
    
    for activity_type, keywords in type_keywords.items():
        if any(keyword in location_lower for keyword in keywords):
            return activity_type
    
    return 'otro'

def _get_default_duration(activity_type: str) -> float:
    """Obtiene la duración por defecto según el tipo de actividad"""
    durations = {
        'desplazamiento': 2.0,  
        'museo': 1.5,
        'restaurante': 1.2,
        'parque': 1.0,
        'monumento': 0.5,
        'hotel': 0.2,
        'playa': 2.0,
        'centro_comercial': 1.5,
        'teatro': 2.0,
        'zoo': 2.5,
        'otro': 1.0
    }
    return durations.get(activity_type, 1.0)


def run_simulation_replicas(simulation_agent, simulation_json: dict, num_replicas: int = 30) -> str:
    """
    Ejecuta múltiples réplicas de simulación del itinerario usando el agente de simulación
    
    Args:
        simulation_agent: Agente de simulación
        simulation_json: JSON estructurado con el itinerario para simular
        num_replicas: Número de réplicas a ejecutar (por defecto 30)
        
    Returns:
        String con los resultados de la simulación formateados
    """
    try:
        print(f"🔍 Iniciando proceso de simulación con {num_replicas} réplicas...")
        print(f"📋 Datos recibidos: {len(simulation_json.get('days', []))} días de itinerario")
        
        
        if not simulation_agent:
            print("⚠️ Agente de simulación no está disponible")
            return ""
        
        
        itinerary_data = []
        context_data = {
            'temporada': simulation_json.get('season', 'verano'),
            'hora_inicio': 9,  
            'prob_lluvia': 0.2,  
            'preferencias_cliente': simulation_json.get('interests', [])  
        }
        
        print(f"🎯 Preferencias del cliente: {context_data['preferencias_cliente']}")
        
        
        for day_info in simulation_json.get('days', []):
            day_num = day_info.get('day', 1)
            day_of_week = day_info.get('day_of_week', 'sabado')
            
            print(f"📅 Procesando día {day_num} ({day_of_week})")
            
            
            if day_num == 1:  
                context_data['dia_semana'] = day_of_week
            
            for i, activity in enumerate(day_info.get('activities', [])):
                
                time_str = activity.get('time', '09:00')
                try:
                    hour = int(time_str.split(':')[0])
                except:
                    hour = 9 + i * 2  
                
                
                is_travel = activity.get('is_travel', False)
                
                place_data = {
                    'nombre': activity.get('location', f'Lugar {i+1}'),
                    'tipo': activity.get('type', 'otro'),
                    'popularidad': activity.get('popularity', 7.0),
                    'distancia_anterior': activity.get('distance_from_previous_km', 2.0) if i > 0 else 0,
                    'distancia_inicio': activity.get('distance_from_previous_km', 3.0) if i == 0 else 0,
                    'dia': day_num,
                    'is_travel': is_travel,
                    'estimated_duration_hours': activity.get('estimated_duration_hours', _get_default_duration(activity.get('type', 'otro'))),
                    'travel_time_hours': activity.get('travel_time_hours', 0.0) if is_travel else 0.0
                }
                
                itinerary_data.append(place_data)
                print(f"  📍 Añadido: {place_data['nombre']} (tipo: {place_data['tipo']})")
        
        
        if not itinerary_data:
            print("⚠️ No hay actividades para simular")
            return ""
        
        print(f"📊 Total de lugares a simular: {len(itinerary_data)}")
        
        
        tourist_profile = simulation_json.get('tourist_profile', 'average')
        
        
        valid_profiles = ['exigente', 'relajado', 'average']
        if tourist_profile not in valid_profiles:
            print(f"⚠️ Perfil '{tourist_profile}' no válido. Usando 'average'")
            tourist_profile = 'average'
        
        
        print(f"🎮 Ejecutando {num_replicas} réplicas de simulación (perfil: {tourist_profile})...")
        
        all_results = []
        successful_simulations = 0
        
        for replica in range(num_replicas):
            try:
                print(f"🔄 Ejecutando réplica {replica + 1}/{num_replicas}...")
                
                simulation_response = simulation_agent.receive({
                    'type': 'simulate_itinerary',
                    'itinerary': itinerary_data,
                    'context': context_data,
                    'profile': tourist_profile
                }, None)  
                
                if simulation_response.get('type') == 'simulation_results':
                    results = simulation_response.get('results', {})
                    all_results.append(results)
                    successful_simulations += 1
                    
                    
                    if (replica + 1) % 5 == 0:
                        satisfaction = results.get('satisfaccion_general', 0)
                        print(f"   ✅ Réplica {replica + 1} completada - Satisfacción: {satisfaction}/10")
                else:
                    error_msg = simulation_response.get('msg', 'Error desconocido')
                    print(f"   ❌ Error en réplica {replica + 1}: {error_msg}")
                    
            except Exception as e:
                print(f"   ❌ Error en réplica {replica + 1}: {e}")
                continue
        
        print(f"🏁 Simulación completada: {successful_simulations}/{num_replicas} réplicas exitosas")
        
        if successful_simulations == 0:
            print("❌ No se pudo completar ninguna simulación")
            return ""
        
        
        aggregated_results = aggregate_simulation_results(all_results)
        
        
        simulation_summary = format_aggregated_simulation_results(aggregated_results, successful_simulations)
        
        
        try:
            simulation_agent.visualizar_resultados(aggregated_results)
            print("📊 Gráficos de simulación agregados guardados en 'simulacion_turista.png'")
        except Exception as e:
            print(f"⚠️ No se pudieron generar los gráficos: {e}")
            import traceback
            traceback.print_exc()
        
        return simulation_summary
            
    except Exception as e:
        print(f"❌ Error ejecutando simulación: {e}")
        import traceback
        traceback.print_exc()
        return ""


def aggregate_simulation_results(all_results: list) -> dict:
    """
    Agrega los resultados de múltiples réplicas de simulación

    Args:
        all_results: Lista de diccionarios con resultados de cada réplica

    Returns:
        Diccionario con resultados agregados
    """
    if not all_results:
        return {}


    satisfacciones = [r.get('satisfaccion_general', 0) for r in all_results]
    cansan_final = [r.get('cansancio_final', 0) for r in all_results]
    duraciones = [r.get('duracion_total_hrs', 0) for r in all_results]
    dias_simulados = [r.get('dias_simulados', 1) for r in all_results]

    aggregated = {
        'perfil_turista': all_results[0].get('perfil_turista', 'average'),
        'num_replicas': len(all_results),


        'satisfaccion_promedio': statistics.mean(satisfacciones),
        'satisfaccion_mediana': statistics.median(satisfacciones),
        'satisfaccion_min': min(satisfacciones),
        'satisfaccion_max': max(satisfacciones),
        'satisfaccion_desv_std': statistics.stdev(satisfacciones) if len(satisfacciones) > 1 else 0,


        'cansancio_promedio': statistics.mean(cansan_final),
        'cansancio_mediana': statistics.median(cansan_final),
        'cansancio_min': min(cansan_final),
        'cansancio_max': max(cansan_final),
        'cansancio_desv_std': statistics.stdev(cansan_final) if len(cansan_final) > 1 else 0,


        'duracion_promedio': statistics.mean(duraciones),
        'duracion_mediana': statistics.median(duraciones),
        'duracion_min': min(duraciones),
        'duracion_max': max(duraciones),


        'dias_promedio': statistics.mean(dias_simulados),
        'dias_max': max(dias_simulados),


        'satisfaccion_general': statistics.mean(satisfacciones),
        'cansancio_final': statistics.mean(cansan_final),
        'duracion_total_hrs': statistics.mean(duraciones),
        'dias_simulados': max(dias_simulados),
    }


    all_places = []
    lugares_por_dia_agregado = {}
    all_comments = []

    for result in all_results:
        places = result.get('lugares_visitados', [])
        all_places.extend(places)


        for place in places:
            comment = place.get('comentario', '')
            if comment and comment != 'Sin comentarios':
                all_comments.append(comment)

        lugares_por_dia = result.get('lugares_por_dia', {})
        for dia, lugares in lugares_por_dia.items():
            if dia not in lugares_por_dia_agregado:
                lugares_por_dia_agregado[dia] = []
            lugares_por_dia_agregado[dia].extend(lugares)


    if all_places:
        place_satisfactions = [p.get('satisfaccion', 0) for p in all_places]
        place_wait_times = [p.get('tiempo_espera_min', 0) for p in all_places]

        aggregated.update({
            'total_visitas': len(all_places),
            'satisfaccion_lugares_promedio': statistics.mean(place_satisfactions),
            'tiempo_espera_promedio': statistics.mean(place_wait_times),
            'lugares_visitados': all_places,
            'lugares_por_dia': lugares_por_dia_agregado
        })


    mejor_experiencia_comentario = ""
    mejor_lugar_nombre = ""
    
    if all_places:

        places_with_comments = [p for p in all_places if p.get('comentario', '') and p.get('comentario', '') != 'Sin comentarios']
        if places_with_comments:
            best_place = max(places_with_comments, key=lambda x: x.get('satisfaccion', 0))
            mejor_experiencia_comentario = best_place.get('comentario', '')
            mejor_lugar_nombre = best_place.get('lugar', '')
    
    aggregated['mejor_experiencia_comentario'] = mejor_experiencia_comentario
    aggregated['mejor_lugar_nombre'] = mejor_lugar_nombre

    satisfaccion_prom = aggregated['satisfaccion_promedio']
    desv_std = aggregated['satisfaccion_desv_std']

    if satisfaccion_prom >= 8:
        if desv_std < 1:
            valoracion = f"¡Experiencia consistentemente excepcional! Con una satisfacción promedio de {satisfaccion_prom:.1f}/10 y baja variabilidad ({desv_std:.1f}), este itinerario ofrece una experiencia turística de alta calidad y confiable."
        else:
            valoracion = f"Experiencia generalmente excelente con satisfacción promedio de {satisfaccion_prom:.1f}/10, aunque con cierta variabilidad ({desv_std:.1f}) entre las experiencias."
    elif satisfaccion_prom >= 6.5:
        valoracion = f"Itinerario satisfactorio con puntuación promedio de {satisfaccion_prom:.1f}/10. La variabilidad de {desv_std:.1f} sugiere que la experiencia puede variar según las condiciones."
    elif satisfaccion_prom >= 5:
        valoracion = f"Experiencia aceptable pero inconsistente. Satisfacción promedio de {satisfaccion_prom:.1f}/10 con alta variabilidad ({desv_std:.1f}) indica necesidad de mejoras."
    else:
        valoracion = f"Itinerario problemático con satisfacción promedio de {satisfaccion_prom:.1f}/10. Se requiere revisión completa del itinerario."

    aggregated['valoracion_viaje'] = valoracion

    return aggregated


def format_aggregated_simulation_results(aggregated_results: dict, num_replicas: int) -> str:
    """
    Formatea los resultados agregados de múltiples réplicas de simulación

    Args:
        aggregated_results: Diccionario con resultados agregados
        num_replicas: Número de réplicas ejecutadas

    Returns:
        String formateado con el resumen de la simulación agregada
    """
    try:
        profile = aggregated_results.get('perfil_turista', 'average')

        summary = f"""
🎮 **SIMULACIÓN DE EXPERIENCIA TURÍSTICA - ANÁLISIS DE {num_replicas} RÉPLICAS**

📊 **Resultados Agregados:**
- Perfil del turista: {profile.capitalize()}
- Réplicas ejecutadas: {num_replicas}

📈 **Satisfacción General:**
- Promedio: {aggregated_results.get('satisfaccion_promedio', 0):.1f}/10 {'⭐' * int(aggregated_results.get('satisfaccion_promedio', 0))}
- Mediana: {aggregated_results.get('satisfaccion_mediana', 0):.1f}/10
- Rango: {aggregated_results.get('satisfaccion_min', 0):.1f} - {aggregated_results.get('satisfaccion_max', 0):.1f}
- Desviación estándar: {aggregated_results.get('satisfaccion_desv_std', 0):.2f}

😴 **Nivel de Cansancio:**
- Promedio: {aggregated_results.get('cansancio_promedio', 0):.1f}/10
- Mediana: {aggregated_results.get('cansancio_mediana', 0):.1f}/10
- Rango: {aggregated_results.get('cansancio_min', 0):.1f} - {aggregated_results.get('cansancio_max', 0):.1f}
- Desviación estándar: {aggregated_results.get('cansancio_desv_std', 0):.2f}

⏱️ **Duración del Viaje:**
- Promedio: {aggregated_results.get('duracion_promedio', 0):.1f} horas
- Rango: {aggregated_results.get('duracion_min', 0):.1f} - {aggregated_results.get('duracion_max', 0):.1f} horas
- Días simulados: {aggregated_results.get('dias_max', 1)}

💭 **Valoración Agregada:**
{aggregated_results.get('valoracion_viaje', 'Sin valoración disponible')}"""

        mejor_comentario = aggregated_results.get('mejor_experiencia_comentario', '')
        mejor_lugar = aggregated_results.get('mejor_lugar_nombre', '')
        if mejor_comentario and mejor_lugar:
            summary += f"\n\n💬 **Mejor experiencia del viaje ({mejor_lugar}):**"
            summary += f"\n\"{mejor_comentario.strip()}\""

        satisfaccion_std = aggregated_results.get('satisfaccion_desv_std', 0)
        cansancio_std = aggregated_results.get('cansancio_desv_std', 0)

        summary += "\n\n🔍 **Análisis de Consistencia:**"

        if satisfaccion_std < 1:
            summary += "\n- ✅ Experiencia muy consistente (baja variabilidad en satisfacción)"
        elif satisfaccion_std < 2:
            summary += "\n- ⚠️ Experiencia moderadamente consistente"
        else:
            summary += "\n- ❌ Experiencia inconsistente (alta variabilidad en satisfacción)"

        if cansancio_std < 1:
            summary += "\n- ✅ Nivel de cansancio predecible"
        else:
            summary += "\n- ⚠️ Nivel de cansancio variable entre réplicas"

        if aggregated_results.get('total_visitas', 0) > 0:
            total_visitas = aggregated_results.get('total_visitas', 0)
            satisfaccion_lugares = aggregated_results.get('satisfaccion_lugares_promedio', 0)
            tiempo_espera = aggregated_results.get('tiempo_espera_promedio', 0)

            summary += f"\n\n📍 **Estadísticas de Lugares:**"
            summary += f"\n- Total de visitas simuladas: {total_visitas}"
            summary += f"\n- Satisfacción promedio por lugar: {satisfaccion_lugares:.1f}/10"
            summary += f"\n- Tiempo de espera promedio: {tiempo_espera:.0f} minutos"
            summary += f"\n- Visitas por réplica: {total_visitas/num_replicas:.1f}"

        summary += "\n\n💡 **Recomendaciones Basadas en el Análisis:**"

        satisfaccion_prom = aggregated_results.get('satisfaccion_promedio', 0)
        cansancio_prom = aggregated_results.get('cansancio_promedio', 0)

        if satisfaccion_prom >= 8 and satisfaccion_std < 1:
            summary += "\n- 🎯 Itinerario óptimo: alta satisfacción y consistente"
        elif satisfaccion_prom >= 7 and satisfaccion_std > 1.5:
            summary += "\n- 🔧 Considerar ajustes para reducir variabilidad"
        elif satisfaccion_prom < 6:
            summary += "\n- ⚠️ Revisar itinerario: satisfacción por debajo del umbral aceptable"

        if cansancio_prom > 8:
            summary += "\n- 😴 Reducir intensidad del itinerario o agregar más descansos"
        elif cansancio_prom < 3:
            summary += "\n- 🚀 Posibilidad de agregar más actividades sin sobrecargar"

        if satisfaccion_std > 2:
            summary += "\n- 📊 Alta variabilidad sugiere factores externos impredecibles"

        satisfaccion_prom = aggregated_results.get('satisfaccion_promedio', 0)
        satisfaccion_std = aggregated_results.get('satisfaccion_desv_std', 0)

        if num_replicas >= 10 and satisfaccion_std > 0:
            error_std = satisfaccion_std / math.sqrt(num_replicas)
            ic_inferior = satisfaccion_prom - 1.96 * error_std
            ic_superior = satisfaccion_prom + 1.96 * error_std

            summary += f"\n\n📊 **Intervalo de Confianza (95%):**"
            summary += f"\n- Satisfacción esperada: {ic_inferior:.1f} - {ic_superior:.1f}/10"

        return summary

    except Exception as e:
        print(f"Error formateando resultados agregados: {e}")
        return f"\n\n⚠️ Error al procesar resultados agregados de {num_replicas} réplicas."


def format_simulation_results(results: dict) -> str:
    """
    Formatea los resultados de una simulación individual en un texto legible

    Args:
        results: Diccionario con los resultados de la simulación

    Returns:
        String formateado con el resumen de la simulación
    """
    try:

        profile = results.get('perfil_turista', 'average')
        general_satisfaction = results.get('satisfaccion_general', 0)
        final_fatigue = results.get('cansancio_final', 0)
        total_duration = results.get('duracion_total_hrs', 0)
        overall_rating = results.get('valoracion_viaje', '')
        places_visited = results.get('lugares_visitados', [])
        dias_simulados = results.get('dias_simulados', 1)
        lugares_por_dia = results.get('lugares_por_dia', {})


        summary = f"""
🎮 **SIMULACIÓN DE EXPERIENCIA TURÍSTICA**

📊 **Resultados Generales:**
- Perfil del turista: {profile.capitalize()}
- Satisfacción general: {general_satisfaction}/10 {'⭐' * int(general_satisfaction)}
- Nivel de cansancio final: {final_fatigue}/10
- Duración total estimada: {total_duration:.1f} horas
- Días simulados: {dias_simulados}

💭 **Valoración del viaje:**
{overall_rating}"""


        if dias_simulados > 1 and lugares_por_dia:
            summary += "\n\n📅 **Experiencia por día:**"
            for dia, lugares in sorted(lugares_por_dia.items()):
                summary += f"\n\n**Día {dia}:**"
                summary += f"\n- Lugares visitados: {len(lugares)}"


                lugares_dia = [p for p in places_visited if p.get('dia', 1) == dia]
                if lugares_dia:
                    avg_satisfaction = sum(p.get('satisfaccion', 0) for p in lugares_dia) / len(lugares_dia)
                    summary += f"\n- Satisfacción promedio: {avg_satisfaction:.1f}/10"


                    best_place = max(lugares_dia, key=lambda x: x.get('satisfaccion', 0))
                    summary += f"\n- Mejor experiencia: {best_place['lugar']} ({best_place['satisfaccion']}/10)"

        summary += "\n\n🏆 **Mejores experiencias del viaje:**"


        if places_visited:
            sorted_places = sorted(places_visited, key=lambda x: x.get('satisfaccion', 0), reverse=True)
            top_places = sorted_places[:3]

            for place in top_places:
                dia_info = f" (Día {place.get('dia', 1)})" if dias_simulados > 1 else ""
                summary += f"\n- {place['lugar']}{dia_info}: {place['satisfaccion']}/10 - {place.get('comentario', 'Sin comentarios')}"


        warnings = []
        if final_fatigue > 8:
            warnings.append("⚠️ El itinerario es muy agotador. Considera reducir actividades o agregar más descansos entre días.")

        if general_satisfaction < 6:
            warnings.append("⚠️ La satisfacción general es baja. Revisa los tiempos de espera y la distribución de actividades.")


        problem_places = [p for p in places_visited if p.get('satisfaccion', 0) < 5]
        if problem_places:
            warnings.append(f"⚠️ {len(problem_places)} lugares con baja satisfacción. Considera alternativas.")


        if lugares_por_dia:
            for dia, lugares in lugares_por_dia.items():
                if len(lugares) > 5:
                    warnings.append(f"⚠️ El día {dia} tiene demasiadas actividades ({len(lugares)}). Considera distribuir mejor.")

        if warnings:
            summary += "\n\n⚠️ **Recomendaciones de mejora:**"
            for warning in warnings:
                summary += f"\n{warning}"


        if places_visited:
            avg_wait_time = sum(p.get('tiempo_espera_min', 0) for p in places_visited) / len(places_visited)
            total_wait_time = sum(p.get('tiempo_espera_min', 0) for p in places_visited)

            summary += f"\n\n📈 **Estadísticas adicionales:**"
            summary += f"\n- Tiempo promedio de espera: {avg_wait_time:.0f} minutos"
            summary += f"\n- Tiempo total en esperas: {total_wait_time:.0f} minutos"
            summary += f"\n- Total de lugares visitados: {len(places_visited)}"
            summary += f"\n- Promedio de lugares por día: {len(places_visited)/dias_simulados:.1f}"

        return summary

    except Exception as e:
        print(f"Error formateando resultados de simulación: {e}")
        return "\n\n⚠️ No se pudieron procesar los resultados de la simulación."
