from faiss import IndexFlatIP, IndexFlatL2


class Retriever:
    def __init__(self, num_neigb,
                 search_methods='IndexFlatL2'):
        self.search_methods = search_methods
        self.num_neigb = num_neigb
        self.time = 0  # store searching time

    def get_similar(self, fea_q, fea_db, file_names=None):
        """ Search similar items from database.

        Args:
            fea_q: Query feature.
            fea_db: Database features.
            file_names(optional): File names of each items in database.

        Returns:
            dist: Distance between the query feature and
                  each of the retrieved features.
            ranked_list: Ranked list of the similar items.
                         If file_names is given, return the names,
                         else, return the index.
            dissm: Dissimilarity.

        """
        data_len, fea_len = fea_db.shape

        dist, ranked_list, dissm = self.search_indexflat(
            fea_q, fea_db, fea_len, self.num_neigb)

        if file_names:
            ranked_names = [file_names[i] for i in ranked_list[0]]
            ranked_list = ranked_names

        return dist, ranked_list, dissm

    def search_indexflat(self, fea_q, fea_db, dim, k):
        index = eval(self.search_methods)(dim)
        index.add(fea_db)
        D, I = index.search(fea_q, k)
        # compute dis-similarity score
        # max_dist = D[method].max() + 1e-10 # TODO: modify this
        dissm = D   # / max_dist
        # TODO: add threshold for the retrieval results
        return D, I, dissm
