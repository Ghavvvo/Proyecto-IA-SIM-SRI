from typing import List, Dict, Any, Optional, Tuple
import google.generativeai as genai
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import warnings
warnings.filterwarnings('ignore')

# Intentar importar dependencias opcionales
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️ sentence-transformers no disponible. Usando TF-IDF como fallback.")

# Configuración de la API de Gemini
GEMINI_API_KEY = "AIzaSyDmW-QXAeksN6hacpCMVpTQnOEAD8MLG00"
genai.configure(api_key=GEMINI_API_KEY)

class RAGSystem:
    def __init__(self, chroma_collection):
        self.collection = chroma_collection
        # Crear instancia del modelo generativo
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def retrieve(self, query: str, top_k: int = 20) -> List[str]:
        """
        Recupera los fragmentos más relevantes para la consulta del usuario.
        Args:
            query (str): Consulta del usuario.
            top_k (int): Número de fragmentos relevantes a recuperar.
        Returns:
            List[str]: Lista de textos relevantes.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        return [doc for doc in results['documents'][0]]

    def generate(self, query: str, context: List[str]) -> str:
        """
        Genera una respuesta basada en la consulta y el contexto recuperado.
        
        IMPORTANTE: Este método SOLO usa información de la base de datos local.
        No consulta fuentes externas ni inventa información.
        
        Args:
            query (str): Consulta del usuario.
            context (List[str]): Fragmentos relevantes recuperados de la BD local.
        Returns:
            str: Respuesta generada por el modelo basada ÚNICAMENTE en la BD.
        """
        # Combinar el contexto en un solo texto
        context_text = "\n".join(context)
        
        prompt = f"""Eres un asistente de turismo. Responde ÚNICAMENTE basándote en la información proporcionada.

Consulta del usuario: {query}

Información disponible en la base de datos LOCAL:
{context_text}

INSTRUCCIONES IMPORTANTES:
- Solo usa la información proporcionada arriba (que viene de la base de datos local)
- NO inventes ni agregues información que no esté explícitamente en el contexto
- Responde siempre proporcionando una lista enumerada de elementos a menos que no tenga sentido proporcionar una lista
- Si no tienes información suficiente, di claramente "No tengo suficiente información sobre este tema en la base de datos"
- Sé conciso y útil
- Si hay información relevante, proporciona las mejores recomendaciones basadas SOLO en los datos disponibles

Respuesta:"""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error al generar contenido: {e}")
            return "Error al procesar la consulta. Por favor, intenta nuevamente."

    def rag_query(self, query: str) -> str:
        """
        Implementa el flujo completo de RAG: recuperación y generación.
        Args:
            query (str): Consulta del usuario.
        Returns:
            str: Respuesta generada.
        """
        # Recuperar contexto directamente de la consulta del usuario
        context = self.retrieve(query)
        
        # Generar respuesta basada en el contexto recuperado
        return self.generate(query, context)


class GeneticDocumentOptimizer:
    """Optimizador genético para selección de documentos usando similitud coseno"""
    
    def __init__(self, population_size: int = 30, generations: int = 20, mutation_rate: float = 0.1, 
                 crossover_rate: float = 0.8, elite_size: int = 2):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        
        print(f"🧬 Optimizador genético configurado:")
        print(f"   • Población: {population_size}")
        print(f"   • Generaciones: {generations}")
        print(f"   • Tasa de mutación: {mutation_rate}")
        print(f"   • Tasa de cruzamiento: {crossover_rate}")
    
    def create_individual(self, total_documents: int, target_size: int) -> List[int]:
        """Crea un individuo (selección de documentos) aleatorio"""
        if total_documents <= target_size:
            return list(range(total_documents))
        return sorted(random.sample(range(total_documents), target_size))
    
    def fitness_function(self, individual: List[int], documents: List[str], 
                        query_embedding: np.ndarray, doc_embeddings: List[np.ndarray]) -> float:
        """
        Calcula la fitness de un individuo basada en similitud coseno con la query
        
        Args:
            individual: Lista de índices de documentos seleccionados
            documents: Lista de todos los documentos
            query_embedding: Embedding de la consulta
            doc_embeddings: Embeddings de todos los documentos
        
        Returns:
            float: Score de fitness (mayor es mejor)
        """
        if not individual:
            return 0.0
        
        # Obtener embeddings de documentos seleccionados
        selected_embeddings = [doc_embeddings[i] for i in individual]
        
        # Calcular similitud coseno promedio con la query
        cosine_scores = []
        for embedding in selected_embeddings:
            similarity = cosine_similarity([query_embedding], [embedding])[0][0]
            cosine_scores.append(similarity)
        
        avg_cosine = np.mean(cosine_scores) if cosine_scores else 0.0
        
        # Calcular diversidad entre documentos seleccionados
        diversity_score = 0.0
        if len(selected_embeddings) > 1:
            similarities = []
            for i in range(len(selected_embeddings)):
                for j in range(i + 1, len(selected_embeddings)):
                    sim = cosine_similarity([selected_embeddings[i]], [selected_embeddings[j]])[0][0]
                    similarities.append(sim)
            # Penalizar alta similitud entre documentos (queremos diversidad)
            diversity_score = 1.0 - np.mean(similarities) if similarities else 0.0
        
        # Calcular score de longitud (preferir contexto de tamaño adecuado)
        selected_docs = [documents[i] for i in individual]
        total_length = sum(len(doc) for doc in selected_docs)
        optimal_length = 2000  # Longitud óptima del contexto
        length_score = 1.0 - abs(total_length - optimal_length) / optimal_length
        length_score = max(0.0, length_score)
        
        # Combinar métricas con pesos
        fitness = (0.7 * avg_cosine +      # 70% similitud con query
                  0.2 * diversity_score +  # 20% diversidad
                  0.1 * length_score)      # 10% longitud óptima
        
        return max(0.0, fitness)
    
    def tournament_selection(self, population: List[List[int]], fitness_scores: List[float], 
                           tournament_size: int = 3) -> List[int]:
        """Selección por torneo"""
        tournament_indices = random.sample(range(len(population)), 
                                         min(tournament_size, len(population)))
        winner_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
        return population[winner_idx].copy()
    
    def crossover(self, parent1: List[int], parent2: List[int], total_documents: int) -> Tuple[List[int], List[int]]:
        """Cruzamiento uniforme entre dos padres"""
        if len(parent1) == 0 or len(parent2) == 0:
            return parent1.copy(), parent2.copy()
        
        # Obtener todos los documentos únicos de ambos padres
        all_docs = sorted(set(parent1 + parent2))
        
        # Crear hijos mediante cruzamiento uniforme
        child1, child2 = [], []
        
        for doc_idx in all_docs:
            if random.random() < 0.5:
                if doc_idx in parent1:
                    child1.append(doc_idx)
                if doc_idx in parent2:
                    child2.append(doc_idx)
            else:
                if doc_idx in parent2:
                    child1.append(doc_idx)
                if doc_idx in parent1:
                    child2.append(doc_idx)
        
        # Asegurar que no excedan el tamaño objetivo
        target_size = max(len(parent1), len(parent2))
        if len(child1) > target_size:
            child1 = sorted(random.sample(child1, target_size))
        if len(child2) > target_size:
            child2 = sorted(random.sample(child2, target_size))
        
        return sorted(child1), sorted(child2)
    
    def mutate(self, individual: List[int], total_documents: int) -> List[int]:
        """Mutación de un individuo"""
        if random.random() > self.mutation_rate:
            return individual
        
        mutated = individual.copy()
        
        # Tipo de mutación aleatoria
        mutation_type = random.choice(['add', 'remove', 'replace'])
        
        if mutation_type == 'add' and len(mutated) < total_documents:
            # Agregar un documento aleatorio no seleccionado
            available = [i for i in range(total_documents) if i not in mutated]
            if available:
                mutated.append(random.choice(available))
        
        elif mutation_type == 'remove' and len(mutated) > 1:
            # Eliminar un documento aleatorio
            mutated.remove(random.choice(mutated))
        
        elif mutation_type == 'replace' and len(mutated) > 0:
            # Reemplazar un documento por otro
            available = [i for i in range(total_documents) if i not in mutated]
            if available:
                old_doc = random.choice(mutated)
                new_doc = random.choice(available)
                mutated.remove(old_doc)
                mutated.append(new_doc)
        
        return sorted(mutated)
    
    def optimize(self, documents: List[str], query_embedding: np.ndarray, 
                doc_embeddings: List[np.ndarray], target_size: int = 8) -> Tuple[List[int], Dict[str, Any]]:
        """
        Ejecuta el algoritmo genético para optimizar la selección de documentos
        
        Returns:
            Tuple[List[int], Dict[str, Any]]: Índices de documentos seleccionados y métricas
        """
        total_documents = len(documents)
        
        if total_documents <= target_size:
            return list(range(total_documents)), {
                'generations_run': 0,
                'best_fitness': 1.0,
                'improvement': 0.0
            }
        
        print(f"🧬 Iniciando optimización genética...")
        print(f"   • Documentos disponibles: {total_documents}")
        print(f"   • Documentos objetivo: {target_size}")
        
        # Inicializar población
        population = []
        for _ in range(self.population_size):
            individual = self.create_individual(total_documents, target_size)
            population.append(individual)
        
        # Evolución
        best_fitness_history = []
        initial_fitness = 0.0
        
        for generation in range(self.generations):
            # Evaluar fitness de toda la población
            fitness_scores = []
            for individual in population:
                fitness = self.fitness_function(individual, documents, query_embedding, doc_embeddings)
                fitness_scores.append(fitness)
            
            # Registrar mejor fitness
            best_fitness = max(fitness_scores)
            best_fitness_history.append(best_fitness)
            
            if generation == 0:
                initial_fitness = best_fitness
            
            # Mostrar progreso cada 5 generaciones
            if generation % 5 == 0:
                avg_fitness = np.mean(fitness_scores)
                print(f"   • Generación {generation}: Mejor={best_fitness:.3f}, Promedio={avg_fitness:.3f}")
            
            # Selección de élite
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            elite = [population[i].copy() for i in elite_indices]
            
            # Crear nueva población
            new_population = elite.copy()  # Mantener élite
            
            # Generar resto de la población
            while len(new_population) < self.population_size:
                # Selección de padres
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                
                # Cruzamiento
                if random.random() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2, total_documents)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutación
                child1 = self.mutate(child1, total_documents)
                child2 = self.mutate(child2, total_documents)
                
                # Agregar hijos a nueva población
                new_population.extend([child1, child2])
            
            # Mantener tamaño de población
            population = new_population[:self.population_size]
        
        # Evaluación final y selección del mejor
        final_fitness_scores = []
        for individual in population:
            fitness = self.fitness_function(individual, documents, query_embedding, doc_embeddings)
            final_fitness_scores.append(fitness)
        
        best_idx = np.argmax(final_fitness_scores)
        best_individual = population[best_idx]
        final_best_fitness = final_fitness_scores[best_idx]
        
        improvement = ((final_best_fitness - initial_fitness) / initial_fitness * 100) if initial_fitness > 0 else 0
        
        print(f"🎯 Optimización completada:")
        print(f"   • Fitness inicial: {initial_fitness:.3f}")
        print(f"   • Fitness final: {final_best_fitness:.3f}")
        print(f"   • Mejora: {improvement:+.1f}%")
        print(f"   • Documentos seleccionados: {len(best_individual)}")
        
        metrics = {
            'generations_run': self.generations,
            'best_fitness': final_best_fitness,
            'initial_fitness': initial_fitness,
            'improvement': improvement,
            'fitness_history': best_fitness_history,
            'population_size': self.population_size
        }
        
        return best_individual, metrics


class EnhancedRAGSystem(RAGSystem):
    """Sistema RAG mejorado con algoritmo genético, distancia coseno y métricas avanzadas"""
    
    def __init__(self, chroma_collection, embedding_model: str = 'TF-IDF', 
                 enable_genetic_optimization: bool = True, genetic_config: Dict[str, Any] = None):
        super().__init__(chroma_collection)
        
        # Configurar modelo de embeddings para métricas
        self.embedding_model_name = embedding_model
        if SENTENCE_TRANSFORMERS_AVAILABLE and embedding_model != 'TF-IDF':
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                self.use_sentence_transformers = True
                print(f"✅ Usando Sentence Transformers: {embedding_model}")
            except:
                print(f"⚠️ Error cargando {embedding_model}, usando TF-IDF")
                self.embedding_model = TfidfVectorizer(max_features=384, stop_words='english')
                self.use_sentence_transformers = False
        else:
            self.embedding_model = TfidfVectorizer(max_features=384, stop_words='english')
            self.use_sentence_transformers = False
            print("✅ Usando TF-IDF para embeddings")
        
        self.tfidf_fitted = False
        
        # Configurar optimizador genético
        self.enable_genetic_optimization = enable_genetic_optimization
        if enable_genetic_optimization:
            # Configuración por defecto del algoritmo genético
            default_config = {
                'population_size': 30,
                'generations': 20,
                'mutation_rate': 0.1,
                'crossover_rate': 0.8,
                'elite_size': 2
            }
            
            # Usar configuración personalizada si se proporciona
            if genetic_config:
                default_config.update(genetic_config)
            
            self.genetic_optimizer = GeneticDocumentOptimizer(**default_config)
            print("✅ Optimización genética habilitada")
        else:
            self.genetic_optimizer = None
            print("⚠️ Optimización genética deshabilitada")
    
    def preprocess_text(self, text: str) -> str:
        """Preprocesa el texto"""
        # Limpiar texto
        text = re.sub(r'\s+', ' ', text)  # Normalizar espacios
        text = re.sub(r'[^\w\s\.\,\!\?\;\:]', '', text)  # Mantener puntuación básica
        return text.strip()
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Crea embeddings para una lista de textos"""
        if self.use_sentence_transformers:
            return self.embedding_model.encode(texts)
        else:
            # Usar TF-IDF
            if not self.tfidf_fitted:
                # Ajustar el vectorizador con todos los textos
                all_texts = texts.copy()
                # Agregar algunos textos adicionales para mejor ajuste del vocabulario
                all_texts.extend([
                    "turismo viaje destino ciudad museo",
                    "hotel restaurante comida gastronomía",
                    "playa montaña naturaleza actividad",
                    "cultura historia arte arquitectura"
                ])
                self.embedding_model.fit(all_texts)
                self.tfidf_fitted = True
            
            embeddings = self.embedding_model.transform(texts)
            return embeddings.toarray()
    
    def calculate_cosine_similarity(self, query_embedding: np.ndarray, document_embeddings: List[np.ndarray]) -> List[float]:
        """Calcula la similitud coseno entre la consulta y los documentos"""
        similarities = []
        for doc_embedding in document_embeddings:
            similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
            similarities.append(similarity)
        return similarities
    
    def retrieve_enhanced(self, query: str, top_k: int = 10) -> Tuple[List[str], Dict[str, Any]]:
        """Recupera documentos usando distancia coseno y métricas avanzadas"""
        # Obtener documentos de ChromaDB
        results = self.collection.query(
            query_texts=[query],
            n_results=min(50, top_k * 3)  # Obtener más documentos para mejor selección
        )
        
        documents = results['documents'][0] if results['documents'] else []
        
        if not documents:
            return [], {'error': 'No se encontraron documentos'}
        
        # Preprocesar documentos
        processed_docs = [self.preprocess_text(doc) for doc in documents]
        
        # Generar embeddings
        query_embeddings = self.create_embeddings([query])
        query_embedding = query_embeddings[0]
        
        doc_embeddings = self.create_embeddings(processed_docs)
        
        # Calcular similitudes coseno
        similarities = self.calculate_cosine_similarity(query_embedding, doc_embeddings)
        
        # Crear lista de documentos con scores
        doc_scores = list(zip(documents, similarities))
        
        # Ordenar por similitud y seleccionar top-k
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        selected_docs = [doc for doc, score in doc_scores[:top_k]]
        selected_scores = [score for doc, score in doc_scores[:top_k]]
        
        # Métricas
        metrics = {
            'total_documents': len(documents),
            'documents_selected': len(selected_docs),
            'avg_relevance': np.mean(selected_scores) if selected_scores else 0.0,
            'max_relevance': np.max(selected_scores) if selected_scores else 0.0,
            'min_relevance': np.min(selected_scores) if selected_scores else 0.0,
            'cosine_similarity_used': True
        }
        
        return selected_docs, metrics
    
    def retrieve_with_genetic_optimization(self, query: str, top_k: int = 8) -> Tuple[List[str], Dict[str, Any]]:
        """Recupera documentos usando algoritmo genético para optimización"""
        # Obtener documentos de ChromaDB
        results = self.collection.query(
            query_texts=[query],
            n_results=min(50, top_k * 5)  # Obtener más documentos para el algoritmo genético
        )
        
        documents = results['documents'][0] if results['documents'] else []
        
        if not documents:
            return [], {'error': 'No se encontraron documentos'}
        
        # Si hay pocos documentos, usar selección tradicional
        if len(documents) <= top_k:
            return self.retrieve_enhanced(query, top_k)
        
        # Preprocesar documentos
        processed_docs = [self.preprocess_text(doc) for doc in documents]
        
        # Generar embeddings
        query_embeddings = self.create_embeddings([query])
        query_embedding = query_embeddings[0]
        
        doc_embeddings = self.create_embeddings(processed_docs)
        
        # Usar algoritmo genético para optimizar selección
        if self.genetic_optimizer:
            optimal_indices, genetic_metrics = self.genetic_optimizer.optimize(
                documents, query_embedding, doc_embeddings, target_size=top_k
            )
            
            # Obtener documentos seleccionados
            selected_docs = [documents[i] for i in optimal_indices]
            
            # Calcular similitudes para métricas
            selected_embeddings = [doc_embeddings[i] for i in optimal_indices]
            similarities = self.calculate_cosine_similarity(query_embedding, selected_embeddings)
            
            # Métricas combinadas
            metrics = {
                'total_documents': len(documents),
                'documents_selected': len(selected_docs),
                'avg_relevance': np.mean(similarities) if similarities else 0.0,
                'max_relevance': np.max(similarities) if similarities else 0.0,
                'min_relevance': np.min(similarities) if similarities else 0.0,
                'genetic_optimization_used': True,
                'genetic_metrics': genetic_metrics
            }
            
            return selected_docs, metrics
        else:
            # Fallback a selección tradicional si no hay optimizador genético
            return self.retrieve_enhanced(query, top_k)
    
    def generate_enhanced(self, query: str, documents: List[str], metrics: Dict[str, Any]) -> str:
        """Genera respuesta mejorada usando los documentos seleccionados"""
        if not documents:
            return "No se encontró información relevante para responder a tu consulta."
        
        # Preparar contexto con información de calidad
        context_text = "\n\n".join(documents)
        
        prompt = f"""Eres un asistente de turismo experto. Analiza cuidadosamente la información proporcionada y responde de manera precisa y útil.

Consulta del usuario: {query}

Información disponible (seleccionada por relevancia):
{context_text}

INSTRUCCIONES:
- Solo usa la información proporcionada arriba
- Responde siempre proporcionando una lista enumerada de elementos a menos que no tenga sentido proporcionar una lista
- Siempre intenta proporcionar una respuesta con la información proporcionada arriba, si no tienes nada de información sobre el tema di claramente "No tengo suficiente información sobre este tema"
- No inventes información que no esté en el contexto
- Sé conciso y útil
- Si hay información relevante, proporciona las mejores recomendaciones basadas en los datos disponibles

Respuesta:"""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error al generar contenido: {e}")
            return "Error al procesar la consulta. Por favor, intenta nuevamente."
    
    def rag_query_enhanced(self, query: str, top_k: int = 10, use_genetic: bool = None) -> Dict[str, Any]:
        """Implementa el flujo completo de RAG mejorado"""
        print(f"🔍 Procesando consulta mejorada: {query}")
        
        # Determinar si usar algoritmo genético
        if use_genetic is None:
            use_genetic = self.enable_genetic_optimization
        
        # Recuperar documentos optimizados
        if use_genetic and self.genetic_optimizer:
            selected_docs, metrics = self.retrieve_with_genetic_optimization(query, top_k)
        else:
            selected_docs, metrics = self.retrieve_enhanced(query, top_k)
        
        if not selected_docs:
            return {
                'query': query,
                'response': "No se encontró información relevante para tu consulta.",
                'metrics': metrics
            }
        
        # Generar respuesta
        response = self.generate_enhanced(query, selected_docs, metrics)
        
        optimization_type = "genético" if metrics.get('genetic_optimization_used', False) else "coseno"
        print(f"✅ Respuesta generada usando {len(selected_docs)} documentos (optimización {optimization_type}, relevancia promedio: {metrics['avg_relevance']:.3f})")
        
        return {
            'query': query,
            'response': response,
            'metrics': metrics,
            'document_details': [
                {
                    'length': len(doc),
                    'preview': doc[:100] + "..." if len(doc) > 100 else doc
                }
                for doc in selected_docs
            ]
        }
    
    # Mantener compatibilidad con la interfaz original
    def rag_query(self, query: str, enhanced: bool = False, **kwargs) -> str:
        """
        Implementa el flujo completo de RAG con opción mejorada.
        Args:
            query (str): Consulta del usuario.
            enhanced (bool): Si usar el sistema mejorado o el tradicional.
        Returns:
            str: Respuesta generada.
        """
        if enhanced:
            result = self.rag_query_enhanced(query, **kwargs)
            return result['response']
        else:
            # Usar el método tradicional
            return super().rag_query(query)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sistema"""
        return {
            'embedding_model': self.embedding_model_name,
            'sentence_transformers_available': SENTENCE_TRANSFORMERS_AVAILABLE,
            'using_sentence_transformers': self.use_sentence_transformers,
            'cosine_similarity_enabled': True,
            'chunking_enabled': False,
            'genetic_optimization_enabled': False
        }