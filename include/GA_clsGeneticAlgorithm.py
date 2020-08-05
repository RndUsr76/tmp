import random
import pandas as pd
import numpy as np
import copy

import include.GA_clsData as DA
import include.GA_clsModel as MO

class Gene(object):
    def __init__(self, _name, _range):
        self.name = _name
        self.value_range = _range
        self.value = self.pickRandomValue()
    
    def pickRandomValue(self):
        return random.choice(self.value_range)

class Individual(object):    
    def __init__(self, genes, params):
        self.fitness = []
        self.genes = []
        self.genes_dict = []
        self.params = params
        self.data = []
        self.model = []
        
        for  gene in genes:
            self.genes.append(Gene(gene[0],gene[1]))
        
        self.genes_dict = self.gene_dict()

    def calculateFitness(self):
        genes_dict = self.genes_dict

        data = DA.Data(genes_dict, self.params)
        
        accs = []
        for _ in range(5):
            data.getDataSet()
            data.getXy(number_validations = 5000)

            model = MO.Model(genes_dict, self.params, data)
            acc_tmp = model.train()
            #print(f'Iteration {i+1}: {acc_tmp}')
            accs.append(acc_tmp)

        fitness = np.float(f'{np.average(accs):0.1f}')
        print(f'Average: {fitness}')
        self.fitness = fitness
        self.updateHyperparameterLog()

        


    def saveIndividual(self, file_name):
        import pickle
        with open('models/' + file_name + '.sav', 'wb') as fw:
            pickle.dump(self, fw)
    
    def printFitness(self):
        print(f'Fitness: {self.fitness}')    
    
    def printGenes(self):
        for g in self.genes_dict:
            print(f'{g}: {self.genes_dict[g]}')
            #print(f'{g.name}: {g.value}')

    def gene_dict(self):
        gdict = {}
        for g in self.genes:
            gdict[g.name] = g.value
        return gdict

    def get_geneNames(self):
        out = []
        for gene in self.genes_dict:
            out.append(gene)
        out.append('fitness')
        return out       

    def get_geneValues(self):
        out = []
        for gene in self.genes_dict:
            out.append(self.genes_dict[gene])
        out.append('{:.1f}'.format(self.fitness))
        return out        

    def updateHyperparameterLog(self):
        FNAME = self.params['hyperparameter_performance_logfile']
        #print('Writing to log file: {}'.format(FNAME))
        try:
            df_parameters = pd.read_csv(FNAME)
        except:
            df_parameters = pd.DataFrame(columns=self.get_geneNames())

        df_parameters = df_parameters.append(pd.DataFrame(columns=df_parameters.columns,data=[self.get_geneValues()]))

        df_parameters.to_csv(FNAME,index=False)


class Population(object):    
    def __init__(self, genes, params):
        self.genes = genes
        self.params = params
        self.individuals = []

        if params['start_new']:  # Initiate random population
            for _ in range(params['initial_population']):
                ind = Individual(self.genes, self.params)
                try:
                    ind.calculateFitness()
                    self.individuals.append(ind)
                except:
                    print('########################### Kunde inte köra denna...')
                    ind.printGenes()
                    print('')
                    print('')
        else: # read data from calc logs and initiate already run population
            df = pd.read_csv(params['hyperparameter_performance_logfile'])

            for _, row in df.iterrows():
                ind = Individual(self.genes, self.params)
                for col in df.columns:
                    if col == 'fitness': continue
                    value = 0
                    if col[0] == 'i': value = int(row[col])
                    if col[0] == 'f': value = float(row[col])
                    if col[0] == 'b': value = bool(row[col])
                    ind.genes_dict[col] = value
                ind.fitness = float(row['fitness'])
                self.individuals.append(ind)

    def sortIndividuals(self):
        self.individuals = sorted(self.individuals,key=lambda individual: individual.fitness, reverse=True)

    def printBestIndividuals(self, number_to_print=-1):
        self.sortIndividuals()
        printed = 0
        print('Top list of individuals')
        for individual in self.individuals:
            individual.printFitness()
            printed += 1
            if number_to_print != -1 and printed >= number_to_print:
                return

    def breedThePopulation(self, generation, mutation_probability = 0.25):
        # Whom to breed
        top_two = True  # Best two individuals
        top_with_random_top10 = True # Best individual with random other individual (among top ten)
        two_random_top10s = True # Two random individual among top 10

        # Breed the top two individuals        
        if top_two:
            self.sortIndividuals()
            parent_a = self.individuals[0]
            parent_b = self.individuals[1]
            
            text = 'Breading top two individuals with fitness: {} and {} with mutation prob: {:.2f} in generation {}'.format(parent_a.fitness, parent_b.fitness, mutation_probability, generation+1)

            try:
                child_1, child_2 = self.breedTwoIndividuals(parent_a,parent_b,mutation_probability, text)
                self.individuals.append(child_1)
                self.individuals.append(child_2)
            except ValueError as e:
                print(f'Did not breed because: {e}')     

        if top_with_random_top10 == True and len(self.individuals)>10:
            self.sortIndividuals()
            parent_a = self.individuals[0]            
            
            ix_b = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
            parent_b = self.individuals[ix_b]
            
            text = 'Breading best individual with random among top ten'
            try:
                child_1, child_2 = self.breedTwoIndividuals(parent_a,parent_b,mutation_probability, text)
                self.individuals.append(child_1)
                self.individuals.append(child_2)
            except ValueError as e:
                print(f'Did not breed because: {e}')
            
        if two_random_top10s == True and len(self.individuals)>10:
            self.sortIndividuals()    
            
            ix_a, ix_b = random.sample(set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 2)
            parent_a = self.individuals[ix_a]
            parent_b = self.individuals[ix_b]

            text = 'Breading random two among top ten'  # but not incl best...
            
            try:
                child_1, child_2 = self.breedTwoIndividuals(parent_a, parent_b, mutation_probability, text)
                self.individuals.append(child_1)
                self.individuals.append(child_2)
            except ValueError as e:
                print(f'Did not breed because: {e}')



    def breedTwoIndividuals(self,i1,i2,mutation_probability, text = ''):
        split_idx = int(len(i1.genes_dict)/2) 
        
        print(text)
        
        # create instance of children
        i3 = Individual(self.genes, self.params)
        i4 = Individual(self.genes, self.params)

        # set genes_dict as mix of parents
        i3.genes[0:split_idx] = copy.deepcopy(i1.genes[0:split_idx])
        i3.genes[split_idx:] = copy.deepcopy(i2.genes[split_idx:])

        i4.genes[0:split_idx] = copy.deepcopy(i2.genes[0:split_idx])
        i4.genes[split_idx:] = copy.deepcopy(i1.genes[split_idx:])

        i3.genes_dict = i3.gene_dict()
        i4.genes_dict = i4.gene_dict()

        # mutate
        def mutate(individual):
            for gene in individual.genes:
                if random.random()<mutation_probability:
                    gene.value = gene.pickRandomValue()
            return individual
        
        i3 = mutate(i3)
        i3.calculateFitness()
                
        i4 = mutate(i4)
        i4.calculateFitness()

        return i3, i4


















