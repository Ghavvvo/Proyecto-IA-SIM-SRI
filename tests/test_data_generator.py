"""
Generador de datos de prueba para comparación de agentes de simulación
Incluye diferentes escenarios de prueba con variedad de condiciones
"""

import random
from typing import Dict, List

class TestDataGenerator:
    """Genera datos de prueba consistentes para evaluar las versiones del agente"""
    
    def __init__(self, seed: int = 42):
        """
        Inicializa el generador con una semilla para reproducibilidad
        
        Args:
            seed: Semilla para el generador aleatorio
        """
        self.seed = seed
        random.seed(seed)
        
        
        self.lugares_base = [
            
            {"nombre": "Museo Nacional de Arte", "tipo": "museo", "popularidad": 8},
            {"nombre": "Museo de Historia Natural", "tipo": "museo", "popularidad": 7},
            {"nombre": "Museo de Arte Moderno", "tipo": "museo", "popularidad": 6},
            
            
            {"nombre": "La Terraza Gourmet", "tipo": "restaurante", "popularidad": 9},
            {"nombre": "Café del Centro", "tipo": "restaurante", "popularidad": 6},
            {"nombre": "Restaurante Tradicional", "tipo": "restaurante", "popularidad": 7},
            
            
            {"nombre": "Parque Central", "tipo": "parque", "popularidad": 8},
            {"nombre": "Jardín Botánico", "tipo": "parque", "popularidad": 7},
            {"nombre": "Parque de la Ciudad", "tipo": "parque", "popularidad": 6},
            
            
            {"nombre": "Monumento Nacional", "tipo": "monumento", "popularidad": 9},
            {"nombre": "Plaza Histórica", "tipo": "monumento", "popularidad": 7},
            {"nombre": "Estatua de la Libertad Local", "tipo": "monumento", "popularidad": 8},
            
            
            {"nombre": "Playa del Sol", "tipo": "playa", "popularidad": 9},
            {"nombre": "Playa Tranquila", "tipo": "playa", "popularidad": 6},
            
            
            {"nombre": "Mall Plaza", "tipo": "centro_comercial", "popularidad": 8},
            {"nombre": "Centro Comercial Downtown", "tipo": "centro_comercial", "popularidad": 7},
            
            
            {"nombre": "Teatro Nacional", "tipo": "teatro", "popularidad": 8},
            {"nombre": "Teatro Municipal", "tipo": "teatro", "popularidad": 6},
            
            
            {"nombre": "Zoológico de la Ciudad", "tipo": "zoo", "popularidad": 8},
            {"nombre": "Acuario Municipal", "tipo": "zoo", "popularidad": 7}
        ]
        
        
        self.perfiles_cliente = [
            {
                "nombre": "Cliente Cultural",
                "preferencias": ["cultura", "historia", "arte", "museums", "monuments"]
            },
            {
                "nombre": "Cliente Gastronómico",
                "preferencias": ["gastronomía", "comida", "restaurants", "food"]
            },
            {
                "nombre": "Cliente Naturaleza",
                "preferencias": ["naturaleza", "aire libre", "parks", "beaches", "nature"]
            },
            {
                "nombre": "Cliente Familiar",
                "preferencias": ["familia", "entretenimiento", "zoo", "parks", "beaches"]
            },
            {
                "nombre": "Cliente Compras",
                "preferencias": ["compras", "shopping", "restaurants", "entertainment"]
            },
            {
                "nombre": "Cliente Mixto",
                "preferencias": ["cultura", "gastronomía", "naturaleza", "compras"]
            }
        ]
        
        
        self.contextos_temporada = [
            {"temporada": "verano", "prob_lluvia": 0.1},
            {"temporada": "invierno", "prob_lluvia": 0.4},
            {"temporada": "primavera", "prob_lluvia": 0.2},
            {"temporada": "otoño", "prob_lluvia": 0.3}
        ]
        
        
        self.dias_semana = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]
    
    def generar_itinerario_corto(self) -> Dict:
        """
        Genera un itinerario corto de 1 día (3-4 lugares)
        
        Returns:
            Diccionario con itinerario y contexto
        """
        num_lugares = random.randint(3, 4)
        lugares_seleccionados = random.sample(self.lugares_base, num_lugares)
        
        itinerario = []
        for i, lugar in enumerate(lugares_seleccionados):
            lugar_copia = lugar.copy()
            lugar_copia["dia"] = 1
            lugar_copia["distancia_anterior"] = random.uniform(1, 5) if i > 0 else 0
            lugar_copia["distancia_inicio"] = random.uniform(2, 8)
            itinerario.append(lugar_copia)
        
        perfil_cliente = random.choice(self.perfiles_cliente)
        contexto = random.choice(self.contextos_temporada).copy()
        contexto.update({
            "dia_semana": random.choice(self.dias_semana),
            "hora_inicio": 9,
            "preferencias_cliente": perfil_cliente["preferencias"],
            "nombre_cliente": perfil_cliente["nombre"]
        })
        
        return {
            "nombre_escenario": f"Itinerario Corto - {perfil_cliente['nombre']}",
            "itinerario": itinerario,
            "contexto": contexto,
            "tipo": "corto"
        }
    
    def generar_itinerario_medio(self) -> Dict:
        """
        Genera un itinerario medio de 2-3 días (6-9 lugares)
        
        Returns:
            Diccionario con itinerario y contexto
        """
        num_dias = random.randint(2, 3)
        lugares_por_dia = random.randint(3, 4)
        total_lugares = num_dias * lugares_por_dia
        
        
        lugares_seleccionados = random.sample(
            self.lugares_base, 
            min(total_lugares, len(self.lugares_base))
        )
        
        itinerario = []
        dia_actual = 1
        lugares_en_dia = 0
        
        for i, lugar in enumerate(lugares_seleccionados):
            lugar_copia = lugar.copy()
            
            
            if lugares_en_dia >= lugares_por_dia:
                dia_actual += 1
                lugares_en_dia = 0
            
            lugar_copia["dia"] = dia_actual
            
            
            if lugares_en_dia == 0:  
                lugar_copia["distancia_anterior"] = 0
                lugar_copia["distancia_inicio"] = random.uniform(3, 10)
            else:
                lugar_copia["distancia_anterior"] = random.uniform(2, 8)
                lugar_copia["distancia_inicio"] = random.uniform(5, 15)
            
            itinerario.append(lugar_copia)
            lugares_en_dia += 1
        
        perfil_cliente = random.choice(self.perfiles_cliente)
        contexto = random.choice(self.contextos_temporada).copy()
        contexto.update({
            "dia_semana": random.choice(self.dias_semana),
            "hora_inicio": 9,
            "preferencias_cliente": perfil_cliente["preferencias"],
            "nombre_cliente": perfil_cliente["nombre"]
        })
        
        return {
            "nombre_escenario": f"Itinerario Medio {num_dias} días - {perfil_cliente['nombre']}",
            "itinerario": itinerario,
            "contexto": contexto,
            "tipo": "medio"
        }
    
    def generar_itinerario_largo(self) -> Dict:
        """
        Genera un itinerario largo de 4-5 días (12-20 lugares)
        
        Returns:
            Diccionario con itinerario y contexto
        """
        num_dias = random.randint(4, 5)
        lugares_por_dia = random.randint(3, 4)
        
        
        itinerario = []
        dia_actual = 1
        lugares_en_dia = 0
        
        for dia in range(1, num_dias + 1):
            
            num_lugares_dia = random.randint(3, 4)
            lugares_dia = random.sample(self.lugares_base, num_lugares_dia)
            
            for i, lugar in enumerate(lugares_dia):
                lugar_copia = lugar.copy()
                lugar_copia["dia"] = dia
                
                
                if dia > 1:
                    lugar_copia["nombre"] = f"{lugar['nombre']} (Día {dia})"
                
                
                if i == 0:  
                    lugar_copia["distancia_anterior"] = 0
                    lugar_copia["distancia_inicio"] = random.uniform(3, 12)
                else:
                    lugar_copia["distancia_anterior"] = random.uniform(2, 10)
                    lugar_copia["distancia_inicio"] = random.uniform(5, 20)
                
                itinerario.append(lugar_copia)
        
        perfil_cliente = random.choice(self.perfiles_cliente)
        contexto = random.choice(self.contextos_temporada).copy()
        contexto.update({
            "dia_semana": random.choice(self.dias_semana),
            "hora_inicio": 9,
            "preferencias_cliente": perfil_cliente["preferencias"],
            "nombre_cliente": perfil_cliente["nombre"]
        })
        
        return {
            "nombre_escenario": f"Itinerario Largo {num_dias} días - {perfil_cliente['nombre']}",
            "itinerario": itinerario,
            "contexto": contexto,
            "tipo": "largo"
        }
    
    def generar_escenario_extremo(self, tipo_extremo: str) -> Dict:
        """
        Genera escenarios extremos para probar límites del sistema
        
        Args:
            tipo_extremo: "optimo" o "pesimo"
            
        Returns:
            Diccionario con itinerario y contexto
        """
        if tipo_extremo == "optimo":
            
            lugares = [l for l in self.lugares_base if l["popularidad"] >= 8][:5]
            
            
            preferencias = []
            for lugar in lugares:
                if lugar["tipo"] == "museo":
                    preferencias.extend(["cultura", "arte"])
                elif lugar["tipo"] == "restaurante":
                    preferencias.extend(["gastronomía", "comida"])
                elif lugar["tipo"] == "playa":
                    preferencias.extend(["playa", "naturaleza"])
            
            contexto = {
                "temporada": "primavera",
                "prob_lluvia": 0.05,
                "dia_semana": "sabado",
                "hora_inicio": 10,
                "preferencias_cliente": list(set(preferencias)),
                "nombre_cliente": "Cliente Ideal"
            }
            
        else:  
            
            lugares = [l for l in self.lugares_base if l["popularidad"] <= 6][:5]
            
            
            preferencias = ["deportes extremos", "vida nocturna", "tecnología"]
            
            contexto = {
                "temporada": "invierno",
                "prob_lluvia": 0.8,
                "dia_semana": "lunes",
                "hora_inicio": 8,
                "preferencias_cliente": preferencias,
                "nombre_cliente": "Cliente Difícil"
            }
        
        
        itinerario = []
        for i, lugar in enumerate(lugares):
            lugar_copia = lugar.copy()
            lugar_copia["dia"] = 1
            lugar_copia["distancia_anterior"] = random.uniform(5, 15) if i > 0 else 0
            lugar_copia["distancia_inicio"] = random.uniform(10, 20)
            itinerario.append(lugar_copia)
        
        return {
            "nombre_escenario": f"Escenario {tipo_extremo.capitalize()}",
            "itinerario": itinerario,
            "contexto": contexto,
            "tipo": f"extremo_{tipo_extremo}"
        }