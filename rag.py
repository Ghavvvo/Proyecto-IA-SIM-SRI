from typing import List
import google.generativeai as genai

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
        Args:
            query (str): Consulta del usuario.
            context (List[str]): Fragmentos relevantes recuperados.
        Returns:
            str: Respuesta generada por el modelo.
        """

        # Combinar el contexto en un solo texto
        context_text = "\n".join(context)
        

        prompt = f"""Eres un asistente de turismo. Responde ÚNICAMENTE basándote en la información proporcionada.

Consulta del usuario: {query}

Información disponible en la base de datos:
{context_text}

INSTRUCCIONES IMPORTANTES:
- Solo usa la información proporcionada arriba
- Responde siempre proporcionando una lista de elementos a menos que no tenga sentido proporcionar una lista
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
