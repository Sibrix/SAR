# versión 1.2

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional, List, Union, Dict
import pickle
import nltk
from SAR_semantics import SentenceBertEmbeddingModel, BetoEmbeddingCLSModel, BetoEmbeddingModel, SpacyStaticModel


## UTILIZAR PARA LA AMPLIACION
# Selecciona un modelo semántico
SEMANTIC_MODEL = "SBERT"
#SEMANTIC_MODEL = "BetoCLS"
#SEMANTIC_MODEL = "Beto"
#SEMANTIC_MODEL = "Spacy"
#SEMANTIC_MODEL = "Spacy_noSW_noA"

def create_semantic_model(modelname):
    assert modelname in ("SBERT", "BetoCLS", "Beto", "Spacy", "Spacy_noSW_noA")
    
    if modelname == "SBERT": return SentenceBertEmbeddingModel()    
    elif modelname == "BetoCLS": return BetoEmbeddingCLSModel()
    elif modelname == "Beto": return BetoEmbeddingModel()
    elif modelname == "Spacy": SpacyStaticModel(remove_stopwords=False, remove_noalpha=False)
    return SpacyStaticModel()


class SAR_Indexer:
    """
    Prototipo de la clase para realizar la indexacion y la recuperacion de artículos de Wikipedia
        
        Preparada para todas las ampliaciones:
          posicionales + busqueda semántica + ranking semántico

    Se deben completar los metodos que se indica.
    Se pueden añadir nuevas variables y nuevos metodos
    Los metodos que se añadan se deberan documentar en el codigo y explicar en la memoria
    """

    # campo que se indexa
    DEFAULT_FIELD = 'all'
    # numero maximo de documento a mostrar cuando self.show_all es False
    SHOW_MAX = 10


    all_atribs = ['urls', 'index', 'docs', 'articles', 'tokenizer', 'show_all',
                  "semantic", "chuncks", "embeddings", "chunck_index", "kdtree", "artid_to_emb"]


    def __init__(self):
        """
        Constructor de la clase SAR_Indexer.
        NECESARIO PARA LA VERSION MINIMA

        Incluye todas las variables necesaria pero
        	puedes añadir más variables si las necesitas. 

        """
        self.urls = set() # hash para las urls procesadas,
        self.index = {} # hash para el indice invertido de terminos --> clave: termino, valor: posting list
        self.docs = {} # diccionario de terminos --> clave: entero(docid),  valor: ruta del fichero.
        self.articles = {} # hash de articulos --> clave entero (artid), valor: la info necesaria para diferencia los artículos dentro de su fichero
        self.tokenizer = re.compile(r"\W+") # expresion regular para hacer la tokenizacion
        self.show_all = False # valor por defecto, se cambia con self.set_showall()

        # PARA LA AMPLIACION
        self.semantic = None
        self.chuncks = []
        self.embeddings = []
        self.chunck_index = []
        self.artid_to_emb = {}
        self.kdtree = None
        self.semantic_threshold = None
        self.semantic_ranking = None # ¿¿ ranking de consultas binarias ??
        self.model = None
        self.MAX_EMBEDDINGS = 200 # número máximo de embedding que se extraen del kdtree en una consulta
        
        
        
        

    ###############################
    ###                         ###
    ###      CONFIGURACION      ###
    ###                         ###
    ###############################


    def set_showall(self, v:bool):
        """

        Cambia el modo de mostrar los resultados.

        input: "v" booleano.

        UTIL PARA TODAS LAS VERSIONES

        si self.show_all es True se mostraran todos los resultados el lugar de un maximo de self.SHOW_MAX, no aplicable a la opcion -C

        """
        self.show_all = v


    def set_semantic_threshold(self, v:float):
        """

        Cambia el umbral para la búsqueda semántica.

        input: "v" booleano.

        UTIL PARA LA AMPLIACIÓN

        si self.semantic es False el umbral no tendrá efecto.

        """
        self.semantic_threshold = v

    def set_semantic_ranking(self, v:bool):
        """

        Cambia el valor de semantic_ranking.

        input: "v" booleano.

        UTIL PARA LA AMPLIACIÓN

        si self.semantic_ranking es True se hará una consulta binaria y los resultados se rankearán por similitud semántica.

        """
        self.semantic_ranking = v


    #############################################
    ###                                       ###
    ###      CARGA Y GUARDADO DEL INDICE      ###
    ###                                       ###
    #############################################


    def save_info(self, filename:str):
        """
        Guarda la información del índice en un fichero en formato binario

        """
        info = [self.all_atribs] + [getattr(self, atr) for atr in self.all_atribs]
        with open(filename, 'wb') as fh:
            pickle.dump(info, fh)

    def load_info(self, filename:str):
        """
        Carga la información del índice desde un fichero en formato binario

        """
        #info = [self.all_atribs] + [getattr(self, atr) for atr in self.all_atribs]
        with open(filename, 'rb') as fh:
            info = pickle.load(fh)
        atrs = info[0]
        for name, val in zip(atrs, info[1:]):
            setattr(self, name, val)


    ###############################
    ###                         ###
    ###   SIMILITUD SEMANTICA   ###
    ###                         ###
    ###############################

            
    def load_semantic_model(self, modelname:str=SEMANTIC_MODEL):
        """
    
        Carga el modelo de embeddings para la búsqueda semántica.
        Solo se debe cargar una vez
        
        """
        if self.model is None:
            print(f"loading {modelname} model ... ",end="", file=sys.stderr)             
            self.model = create_semantic_model(modelname)
            print("done!", file=sys.stderr)

            

    def update_chuncks(self, txt:str, artid:int):
        """
        
        Añade los chuncks (frases en nuestro caso) del texto "txt" correspondiente al articulo "artid" en la lista de chuncks
        Pasos:
            1 - extraer los chuncks de txt, en nuestro caso son las frases. Se debe utilizar "sent_tokenize" de la librería "nltk"
            2 - actualizar los atributos que consideres necesarios: self.chuncks, self.embeddings, self.chunck_index y self.artid_to_emb.
        """
        """
        Añade los chuncks (frases) del texto "txt" correspondiente al articulo "artid".
        """
        # 1 - extraer los chuncks (frases) usando nltk
        import nltk
        try:
            sentences = nltk.sent_tokenize(txt)
        except LookupError:
            # Por si no tienes descargado el paquete de tokenización localmente
            nltk.download('punkt')
            nltk.download('punkt_tab')
            sentences = nltk.sent_tokenize(txt)

        # 2 - actualizar los atributos necesarios
        if artid not in self.artid_to_emb:
            self.artid_to_emb[artid] = []
            
        for sentence in sentences:
            chunk_idx = len(self.chuncks)
            self.chuncks.append(sentence)
            self.chunck_index.append(artid) # El índice de la frase apunta al ID del artículo
            self.artid_to_emb[artid].append(chunk_idx)
              
        

    def create_kdtree(self):
        """
        
        Crea el tktree utilizando un objeto de la librería SAR_semantics
        Solo se debe crear una vez despues de indexar todos los documentos
        
        # 1: Se debe llamar al método fit del modelo semántico
        # 2: Opcionalmente se puede guardar información del modelo semántico (kdtree y/o embeddings) en el SAR_Indexer
        
        """
        print(f"Creating kdtree ...", end="", flush=True)
        # 1: Llamar al método fit del modelo semántico pasándole todas las frases
        self.model.fit(self.chuncks)
        
        # 2: Guardar información del modelo semántico en el SAR_Indexer para que se guarde en disco
        self.kdtree = self.model.kdtree
        self.embeddings = self.model.embeddings
        print("done!")


        
    def solve_semantic_query(self, query:str):
        """

        Resuelve una consulta utilizando el modelo semántico.
        Pasos:
            1 - utiliza el método query del modelo sémantico
            2 - devuelve top_k resultados, inicialmente top_k puede ser MAX_EMBEDDINGS
            3 - si el último resultado tiene una distancia <= self.semantic_threshold 
                  ==> no se han recuperado todos los resultado: vuelve a 2 aumentando top_k
            4 - también se puede salir si recuperamos todos los embeddings
            5 - tenemos una lista de chuncks que se debe pasar a artículos
        """

        self.load_semantic_model()
        
        self.load_semantic_model()
        # Inyectamos en el modelo el árbol y los embeddings que cargamos desde disco
        self.model.set_kdtree(self.kdtree)
        self.model.set_embeddings(self.embeddings)
        
        top_k = self.MAX_EMBEDDINGS
        total_chunks = len(self.chuncks)
        
        # Bucle para aumentar el número de resultados extraídos si no llegamos al umbral
        while True:
            # 1 - método query del modelo sémantico
            results = self.model.query(query, top_k)
            last_dist = results[-1][0]
            
            # 3 - Comprobar umbral y aumentar top_k si es necesario
            if self.semantic_threshold is not None and last_dist <= self.semantic_threshold:
                if top_k >= total_chunks: # 4 - Salir si recuperamos todo
                    break
                top_k = min(top_k * 2, total_chunks)
            else:
                break
                
        # 5 - Convertir lista de chuncks a artículos sin repetir
        final_articles = []
        for dist, idx in results:
            if self.semantic_threshold is not None and dist > self.semantic_threshold:
                continue # Descartar los que están por encima del umbral
                
            artid = self.chunck_index[idx]
            if artid not in final_articles:
                final_articles.append(artid)
                
        return final_articles


    def semantic_reranking(self, query:str, articles: List[int]):
        """

        Ordena los articulos en la lista 'article' por similitud a la consulta 'query'.
        Pasos:
            1 - utiliza el método query del modelo sémantico
            2 - devuelve top_k resultado, inicialmente top_k puede ser MAX_EMBEDDINGS
            3 - a partir de los chuncks se deben obtener los artículos
            3 - si entre los artículos recuperados NO estan todos los obtenidos por la RI binaria
                  ==> no se han recuperado todos los resultado: vuelve a 2 aumentando top_k
            4 - se utiliza la lista ordenada del kdtree para ordenar la lista "articles"
        """
        
        self.load_semantic_model()
        if not articles:
            return []
            
        self.load_semantic_model()
        self.model.set_kdtree(self.kdtree)
        self.model.set_embeddings(self.embeddings)
        
        top_k = self.MAX_EMBEDDINGS
        total_chunks = len(self.chuncks)
        
        target_articles = set(articles)
        recovered_articles = set()
        ordered_articles = []
        
        while True:
            # 1 y 2 - Extraemos top_k chunks más cercanos
            results = self.model.query(query, top_k)
            
            # 3 - Extraemos los artículos de esos chunks
            for dist, idx in results:
                artid = self.chunck_index[idx]
                # Si el artículo está en nuestra lista de "RI binaria" y no lo habíamos procesado
                if artid in target_articles and artid not in recovered_articles:
                    recovered_articles.add(artid)
                    ordered_articles.append(artid) # Se añaden en orden de llegada (similitud)
                    
            # 4 - Si ya hemos recuperado todos los de la búsqueda binaria, salimos
            if len(recovered_articles) == len(target_articles) or top_k >= total_chunks:
                break
                
            # Aumentamos rango
            top_k = min(top_k * 2, total_chunks)
            
        # Si algún artículo quedó rezagado y nunca apareció (muy improbable), lo ponemos al final
        for artid in articles:
            if artid not in recovered_articles:
                ordered_articles.append(artid)
                
        return ordered_articles
    

    ###############################
    ###                         ###
    ###   PARTE 1: INDEXACION   ###
    ###                         ###
    ###############################

    def already_in_index(self, article:Dict) -> bool:
        """

        Args:
            article (Dict): diccionario con la información de un artículo

        Returns:
            bool: True si el artículo ya está indexado, False en caso contrario
        """
        return article['url'] in self.urls


    def index_dir(self, root:str, **args):
        """

        Recorre recursivamente el directorio o fichero "root"
        NECESARIO PARA TODAS LAS VERSIONES

        Recorre recursivamente el directorio "root"  y indexa su contenido
        los argumentos adicionales "**args" solo son necesarios para las funcionalidades ampliadas

        """
        self.positional = args['positional']
        self.semantic = args['semantic']
        if self.semantic is True:
            self.load_semantic_model()


        file_or_dir = Path(root)

        if file_or_dir.is_file():
            # is a file
            self.index_file(root)
        elif file_or_dir.is_dir():
            # is a directory
            for d, _, files in os.walk(root):
                for filename in sorted(files):
                    if filename.endswith('.json'):
                        fullname = os.path.join(d, filename)
                        self.index_file(fullname)
        else:
            print(f"ERROR:{root} is not a file nor directory!", file=sys.stderr)
            sys.exit(-1)

        if self.semantic is True:
            self.create_kdtree()
        #####################################################
        ## COMPLETAR SI ES NECESARIO FUNCIONALIDADES EXTRA ##
        #####################################################
        
        
    def parse_article(self, raw_line:str) -> Dict[str, str]:
        """
        Crea un diccionario a partir de una linea que representa un artículo del crawler

        Args:
            raw_line: una linea del fichero generado por el crawler

        Returns:
            Dict[str, str]: claves: 'url', 'title', 'summary', 'all', 'section-name'
        """
        
        article = json.loads(raw_line)
        sec_names = []
        txt_secs = ''
        for sec in article['sections']:
            txt_secs += sec['name'] + '\n' + sec['text'] + '\n'
            txt_secs += '\n'.join(subsec['name'] + '\n' + subsec['text'] + '\n' for subsec in sec['subsections']) + '\n\n'
            sec_names.append(sec['name'])
            sec_names.extend(subsec['name'] for subsec in sec['subsections'])
        article.pop('sections') # no la necesitamos
        article['all'] = article['title'] + '\n\n' + article['summary'] + '\n\n' + txt_secs
        article['section-name'] = '\n'.join(sec_names)

        return article


    def index_file(self, filename:str):
        """

        Indexa el contenido de un fichero.

        input: "filename" es el nombre de un fichero generado por el Crawler cada línea es un objeto json
            con la información de un artículo de la Wikipedia

        NECESARIO PARA TODAS LAS VERSIONES

        dependiendo del valor de self.positional se debe ampliar el indexado

        """
        if filename not in self.docs.values():
            docid = len(self.docs)
            self.docs[docid] = filename
        else:
            docid = list(self.docs.keys())[list(self.docs.values()).index(filename)]

        for i, line in enumerate(open(filename)):
            j = self.parse_article(line)
            if self.already_in_index(j):
                continue
            self.urls.add(j['url'])
            artid = len(self.articles)
            self.articles[artid] = {'title': j['title'], 'url': j['url']}
            text = j[self.DEFAULT_FIELD]
            # --- Procesamiento semántico por artículo ---
            if self.semantic:
                self.update_chuncks(text, artid)
            tokens = self.tokenize(text)
            for pos,term in enumerate(tokens):
                if term not in self.index:
                    # Si es posicional usamos diccionario {artid: [pos1, pos2]}, 
                    # si no, una lista de artids [artid1, artid2]
                    self.index[term] = {} if self.positional else []
                if self.positional:
                    if artid not in self.index[term]:
                        self.index[term][artid] = []
                    self.index[term][artid].append(pos)
                else:
                   # Añadir solo si el artículo no está ya en la posting list del término
                    if len(self.index[term]) == 0 or self.index[term][-1] != artid:
                        self.index[term].append(artid)
                   

    def tokenize(self, text:str):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Tokeniza la cadena "texto" eliminando simbolos no alfanumericos y dividientola por espacios.
        Puedes utilizar la expresion regular 'self.tokenizer'.

        params: 'text': texto a tokenizar

        return: lista de tokens

        """
        return self.tokenizer.sub(' ', text.lower()).split()




    def show_stats(self):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Muestra estadisticas de los indices

        """
        print("=" * 40)
        print(f"Number of indexed files: {len(self.docs)}")
        print(f"Number of indexed articles: {len(self.articles)}")
        print(f"Number of terms: {len(self.index)}")
        print(f"Positional index: {self.positional}")
        print("=" * 40)



    #################################
    ###                           ###
    ###   PARTE 2: RECUPERACION   ###
    ###                           ###
    #################################

    ###################################
    ###                             ###
    ###   PARTE 2.1: RECUPERACION   ###
    ###                             ###
    ###################################


    def solve_query(self, query:str, prev:Dict={}):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una query.
        Debe realizar el parsing de consulta que sera mas o menos complicado en funcion de la ampliacion que se implementen


        param:  "query": cadena con la query
                "prev": incluido por si se quiere hacer una version recursiva. No es necesario utilizarlo.


        return: posting list con el resultado de la query

        """
        
        if query is None or len(query) == 0:
            return []
        
        # 1. Separamos por espacios usando split() normal
        raw_terms = query.split()
        
        # 2. Reconstruimos los términos uniendo los que van entre comillas
        tokens = []
        in_quotes = False
        current_phrase = []
        for tk in raw_terms:
            if not in_quotes:
                # Si empieza y acaba por comillas (ej: '"hola"')
                if tk.startswith('"') and tk.endswith('"') and len(tk) > 1:
                    tokens.append(tk)
                # Si solo empieza por comillas (ej: '"fin')
                elif tk.startswith('"'):
                    in_quotes = True
                    current_phrase.append(tk)
                # Si es una palabra normal o NOT
                else:
                    tokens.append(tk)
            else:
                # Estamos dentro de unas comillas, seguimos agrupando
                current_phrase.append(tk)
                # Si termina con comillas (ej: 'semana"')
                if tk.endswith('"'):
                    in_quotes = False
                    tokens.append(" ".join(current_phrase))
                    current_phrase = []
        # 3. Ahora procesamos los tokens resultantes (AND, NOT y Búsqueda Posicional)

        result = []
        is_first = True
        is_not = False
        
        for tk in tokens:
            # Gestionamos el flag NOT
            if tk == 'NOT':
                is_not = True
                continue
                
            # Si es una frase entre comillas (Búsqueda posicional)
            if tk.startswith('"') and tk.endswith('"'):
                phrase = tk[1:-1] # Quitamos las comillas
                terms = self.tokenize(phrase)
                
                if len(terms) == 0:
                    continue
                elif len(terms) == 1:
                    current_posting = self.get_posting(terms[0])
                else:
                    current_posting = self.get_positionals(terms)
                    
            # Si es un término normal
            else:
                term = self.tokenize(tk)
                if len(term) > 0:
                    current_posting = self.get_posting(term[0])
                else:
                    continue
            # Si el término actual venía precedido de un NOT, invertimos su posting list
            if is_not:
                current_posting = self.reverse_posting(current_posting)
                is_not = False
            # Si es el primer término, inicializamos el resultado
            if is_first:
                result = current_posting
                is_first = False
            # Cruzamos los resultados con los que ya teníamos calculados
            else:
                result = self.and_posting(result, current_posting)

        # Si se activó -S (búsqueda pura semántica independientemente de los términos)
        if self.semantic_threshold is not None:
            result = self.solve_semantic_query(query)
            
        # Si se activó -R (hacemos intersección booleana normal, y luego ordenamos semánticamente)
        elif self.semantic_ranking:
            result = self.semantic_reranking(query, result)
            
        # ----------------------------------------------------
        
        return result, prev
        




    def get_posting(self, term:str):
        """

        Devuelve la posting list asociada a un termino.
        Puede llamar self.get_positionals: para las búsquedas posicionales.


        param:  "term": termino del que se debe recuperar la posting list.

        return: posting list

        NECESARIO PARA TODAS LAS VERSIONES

        """
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################

        if term in self.index:
            if self.positional:
                return list(self.index[term].keys())
            else:
                return self.index[term]
        return []



    def get_positionals(self, terms:str):
        """
        Devuelve la posting list asociada a una secuencia de terminos consecutivos.
        NECESARIO PARA LAS BÚSQUESAS POSICIONALES

        param:  "terms": lista con los terminos consecutivos para recuperar la posting list.

        return: posting list

        """

        if not self.positional or len(terms) == 0:
            return []
            
        # 1. Comprobamos que todos los términos existan en el índice. 
        # Si uno solo falla, la frase no puede existir.
        for t in terms:
            if t not in self.index:
                return []
                
        # 2. Intersección inicial de artículos (AND).
        # Extraemos las claves (artids) del diccionario del primer término
        common_articles = list(self.index[terms[0]].keys())
        
        # Hacemos un AND secuencial con los artids de los demás términos
        for i in range(1, len(terms)):
            next_articles = list(self.index[terms[i]].keys())
            common_articles = self.and_posting(common_articles, next_articles)
            
            # Si en algún momento la intersección se queda vacía, paramos
            if not common_articles:
                return []
                
        # 3. Comprobación posicional en los artículos candidatos
        result = []
        for artid in common_articles:
            # Obtenemos las listas de posiciones de cada término para este artículo específico
            # Ejemplo: pos_lists = [[2, 15], [3, 20], [4, 88]] (posiciones de "fin", "de", "semana")
            pos_lists = [self.index[t][artid] for t in terms]
            
            match_found = False
            
            # Recorremos las posiciones en las que aparece el PRIMER término
            for p0 in pos_lists[0]:
                is_consecutive = True
                
                # Comprobamos si los siguientes términos están en las posiciones consecutivas (p0+1, p0+2...)
                for i in range(1, len(terms)):
                    # Si la posición esperada no está en la lista de posiciones del término "i"
                    if (p0 + i) not in pos_lists[i]:
                        is_consecutive = False
                        break # Rompemos este bucle interior, esta secuencia no sirve
                
                # Si hemos encontrado al menos una secuencia completa y consecutiva
                if is_consecutive:
                    match_found = True
                    break # No hace falta seguir buscando más secuencias en este mismo artículo
                    
            # Si el artículo superó la prueba posicional, lo añadimos al resultado final
            if match_found:
                result.append(artid)
                
        return result



    def reverse_posting(self, p:list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Devuelve una posting list con todas las noticias excepto las contenidas en p.
        Util para resolver las queries con NOT.


        param:  "p": posting list


        return: posting list con todos los artid exceptos los contenidos en p

        """
        # Obtenemos la lista con TODOS los identificadores de los artículos.
        # Al haberse generado secuencialmente en index_file, ya está ordenada.
        all_articles = list(self.articles.keys())
        
        # El reverse es simplemente: TODOS los artículos EXCEPT los que están en 'p'
        return self.minus_posting(all_articles, p)



    def and_posting(self, p1:list, p2:list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Calcula el AND de dos posting list de forma EFICIENTE

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los artid incluidos en p1 y p2

        """
        #No skip Pointers?
        articles = []
        pi = 0
        pj = 0
        while pi < len(p1) and pj < len(p2):
            if p1[pi] == p2[pj]:
                articles.append(p1[pi])
                pi += 1
                pj += 1
            elif p1[pi] < p2[pj]:
                pi += 1
            else:
                pj += 1

        return articles
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################






    def minus_posting(self, p1, p2):
        """
        OPCIONAL PARA TODAS LAS VERSIONES

        Calcula el except de dos posting list de forma EFICIENTE.
        Esta funcion se incluye por si es util, no es necesario utilizarla.

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los artid incluidos de p1 y no en p2

        """
        result = []
        i = 0
        j = 0
        
        while i < len(p1) and j < len(p2):
            if p1[i] == p2[j]:
                # El artículo está en p1 y en p2 -> Lo excluimos y avanzamos ambos
                i += 1
                j += 1
            elif p1[i] < p2[j]:
                # El artículo está en p1 y es menor que el actual de p2 -> Lo incluimos
                result.append(p1[i])
                i += 1
            else:
                # El artículo de p2 es menor que el de p1 -> Avanzamos p2 para intentar alcanzarlo
                j += 1
                
        # Si hemos terminado de recorrer p2 pero aún quedan elementos en p1,
        # los añadimos todos porque sabemos que no están en p2
        while i < len(p1):
            result.append(p1[i])
            i += 1
            
        return result






    #####################################
    ###                               ###
    ### PARTE 2.2: MOSTRAR RESULTADOS ###
    ###                               ###
    #####################################

    def solve_and_count(self, ql:List[str], verbose:bool=True) -> List:
        results = []
        for query in ql:
            if len(query) > 0 and query[0] != '#':
                r, _ = self.solve_query(query)
                results.append(len(r))
                if verbose:
                    print(f'{query}\t{len(r)}')
            else:
                results.append(0)
                if verbose:
                    print(query)
        return results


    def solve_and_test(self, ql:List[str]) -> bool:
        errors = False
        for line in ql:
            if len(line) > 0 and line[0] != '#':
                query, ref = line.split('\t')
                reference = int(ref)
                result, _ = self.solve_query(query)
                result = len(result)
                if reference == result:
                    print(f'{query}\t{result}')
                else:
                    print(f'>>>>{query}\t{reference} != {result}<<<<')
                    errors = True
            else:
                print(line)

        return not errors


    def solve_and_show(self, query:str):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una consulta y la muestra junto al numero de resultados

        param:  "query": query que se debe resolver.

        return: el numero de artículo recuperadas, para la opcion -T

        """
        """
        Resuelve una consulta y la muestra junto al numero de resultados
        """
        
        result, _ = self.solve_query(query)
        
        print(f"Query: '{query}'")
        print(f"Number of results: {len(result)}")
        
        # Límite de resultados a mostrar según la configuración self.show_all
        limit = len(result) if self.show_all else min(len(result), self.SHOW_MAX)
        
        for i in range(limit):
            artid = result[i]
            article = self.articles[artid]
            title = article.get('title', 'N/A')
            url = article.get('url', 'N/A')
            # Formato de visualización: Orden | artid | Titulo | Url
            print(f"#{i+1}\t({artid})\t{title}\t{url}")
            