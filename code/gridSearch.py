from sklearn.model_selection import ParameterGrid
import inspect
from gan import GAN

def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]

class gridSearch:

    def __init__(self, dataset, parameters):
        self.dataset = dataset
        self.parameters = parameters
        self.name = retrieve_name(parameters)
        self.iter = 0
        self.results = []

    def fit(self):

        for g in ParameterGrid(self.parameters):
            self.iter += 1
            print('\nTraining:', str(self.iter) + '/' + str(len(ParameterGrid(self.parameters))), '- Parameters:', g)

            # Model
            model = GAN(self.name, self.dataset, **g)

            score = model.train()

            print('\tScore: emd: %f\t fid: %f\t inception: %f\t knn: %f\t mmd: %f\t mode: %f' % (score.emd, score.fid, score.inception, score.knn, score.mmd, score.mode))

            self.results.append({'score':score, 'params':g})

            # Write to results
            with open('../resources/results.txt', "a+") as f:
                f.write('emd: %f\t fid: %f\t inception: %f\t knn: %f\t mmd: %f\t mode: %f' % (score.emd, score.fid, score.inception, score.knn, score.mmd, score.mode))

    def summary(self):
        # Summarize results
        print('\nSummary')
        for res in self.results:
            score = vars(res['score'])
            print("Score: %r - Parameters: %r" % (score, res['params']))