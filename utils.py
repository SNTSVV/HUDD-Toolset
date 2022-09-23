from candidate import Candidate
import numpy as np
import random
import sys


def calculate_minimum_distance(candidate, random_pop):
    distance = 1000
    for each_candidate in random_pop:
        vals = each_candidate.get_candidate_values()
        candidate_vals = candidate.get_candidate_values()
        dist = np.linalg.norm(np.array(vals) - np.array(candidate_vals))
        if dist < distance:
            distance = dist
    return distance


def generate_adaptive_random_population(size, lb, ub):
    random_pop = []
    random_pop.append(generate_random_population(1, lb, ub)[0])

    while len(random_pop) < size:
        D = 0
        selected_candidate = None
        rp = generate_random_population(size, lb, ub)
        for each_candidate in rp:
            min_dis = calculate_minimum_distance(each_candidate, random_pop)
            if min_dis > D:
                D = min_dis
                selected_candidate = each_candidate
        random_pop.append(selected_candidate)

    return random_pop


# random value generator
def generate_random_population(size, lb, ub):
    random_pop = []

    for i in range(size):
        candidate_vals = []
        for index in range(len(lb)):
            candidate_vals.append(random.uniform(lb[index], ub[index]))

        random_pop.append(Candidate(candidate_vals))
    return random_pop


# dominates method, same from paper
def dominates(value_from_pop, value_from_archive, objective_uncovered):
    dominates_f1 = False
    dominates_f2 = False
    for each_objective in objective_uncovered:
        f1 = value_from_pop[each_objective]
        f2 = value_from_archive[each_objective]
        if f1 < f2:
            dominates_f1 = True
        if f2 < f1:
            dominates_f2 = True
        if dominates_f1 and dominates_f2:
            break
    if dominates_f1 == dominates_f2:
        return False
    elif dominates_f1:
        return True
    return False


# calling the fitness value function

def evaulate_population(func, pop):
    for candidate in pop:
        result = func(candidate.get_candidate_values())
        candidate.set_objective_values(result)


def exists_in_archive(archive, index):
    for candidate in archive:
        if candidate.exists_in_satisfied(index):
            return True
    return False


# searching archive
def get_from_archive(obj_index, archive):
    for candIndx in range(len(archive)):
        candidate = archive[candIndx]
        if candidate.exists_in_satisfied(obj_index):
            return candidate, candIndx
    return None


# updating archive with adding the number of objective it satisfies [1,2,3,4,[ob1,ob2, objective index]]
def update_archive(pop, objective_uncovered, archive, no_of_Objectives, threshold_criteria):
    for objective_index in range(no_of_Objectives):
        for pop_index in range(len(pop)):
            objective_values = pop[pop_index].get_objective_values()

            if objective_values[objective_index] <= threshold_criteria[objective_index]:
                if exists_in_archive(archive, objective_index):
                    archive_value, cand_indx = get_from_archive(objective_index, archive)
                    obj_archive_values = archive_value.get_objective_values()
                    if obj_archive_values[objective_index] > objective_values[objective_index]:
                        value_to_add = pop[pop_index]
                        value_to_add.add_objectives_covered(objective_index)
                        # archive.append(value_to_add)
                        archive[cand_indx] = value_to_add
                        if objective_index in objective_uncovered:
                            objective_uncovered.remove(objective_index)
                        # archive.remove(archive_value)
                else:
                    value_to_add = pop[pop_index]
                    value_to_add.add_objectives_covered(objective_index)
                    archive.append(value_to_add)
                    if objective_index in objective_uncovered:
                        objective_uncovered.remove(objective_index)


# method to get the most dominating one
def select_best(tournament_candidates, objective_uncovered):
    best = tournament_candidates[0]  # in case none is dominating other
    for i in range(len(tournament_candidates)):
        candidate1 = tournament_candidates[i]
        for j in range(len(tournament_candidates)):
            candidate2 = tournament_candidates[j]
            if (dominates(candidate1.get_objective_values(), candidate2.get_objective_values(), objective_uncovered)):
                best = candidate1
    return best


def tournament_selection_improved(pop, size, objective_uncovered):
    tournament_candidates = []
    for i in range(size):
        indx = random.randint(0, len(pop) - 1)
        random_candidate = pop[indx]
        tournament_candidates.append(random_candidate)

    best = select_best(tournament_candidates, objective_uncovered)
    return best;
def tournament_selection(pop, size, objective_uncovered):
    tournament_candidates = []
    for i in range(size):
        indx = random.randint(0, len(pop) - 1)
        random_candidate = pop[indx]
        tournament_candidates.append(random_candidate)

    best = select_best(tournament_candidates, objective_uncovered)
    return best;


def do_single_point_crossover(parent1, parent2):  # verify with donghwan
    parent1 = parent1.get_candidate_values();
    parent2 = parent2.get_candidate_values()
    crossover_point = random.randint(1, len(parent1) - 1)
    t_parent1 = parent1[0:crossover_point]
    t_parent2 = parent2[0:crossover_point]
    for i in range(crossover_point, len(parent1)):
        t_parent1.append(parent2[i])
        t_parent2.append(parent1[i])

    return Candidate(t_parent1), Candidate(t_parent2)


def do_uniform_mutation(parent1, parent2, lb, threshold):
    child1 = []
    child2 = []

    parent1 = parent1.get_candidate_values();
    parent2 = parent2.get_candidate_values()

    for parent1_index in range(len(parent1)):
        probability_mutation = random.uniform(0, 1)
        if probability_mutation <= threshold:
            random_value = random.uniform(lb[parent1_index], parent1[parent1_index])
            child1.append(random_value)
        else:
            child1.append(parent1[parent1_index])

    for parent2_index in range(len(parent2)):
        probability_mutation = random.uniform(0, 1)
        if probability_mutation <= 0.25:  # 1/4         25% probability
            random_value = random.uniform(lb[parent2_index], parent2[parent2_index])
            child2.append(random_value)
        else:
            child2.append(parent2[parent2_index])

    return Candidate(child1), Candidate(child2)


def generate_off_spring(pop, objective_uncovered, lb):
    size = len(pop)
    population_to_return = []

    while (len(population_to_return) < size):

        parent1 = tournament_selection(pop, 10, objective_uncovered)  # tournament selection same size as paper
        parent2 = tournament_selection(pop, 10, objective_uncovered)
        probability_crossover = random.uniform(0, 1)
        if probability_crossover <= 0.75:  # 75% probability
            parent1, parent2 = do_single_point_crossover(parent1, parent2)  # check with donghwan

        child1, child2 = do_uniform_mutation(parent1, parent2, lb, (1 / len(parent1.get_candidate_values())))
        population_to_return.append(child1)
        population_to_return.append(child2)

    return population_to_return

def generate_off_spring_mosaplus(pop, objective_uncovered, lb):
    size = len(pop)
    population_to_return = []

    while (len(population_to_return) < size):

        parent1 = tournament_selection(pop, 10, objective_uncovered)  # tournament selection same size as paper
        parent2 = tournament_selection(pop, 10, objective_uncovered)
        probability_crossover = random.uniform(0, 1)
        if probability_crossover <= 0.75:  # 75% probability
            nc=get_distribution_index(parent1, parent2, objective_uncovered, []);
            parent1, parent2 = do_simulated_binary_crossover(parent1, parent2,nc)  # check with donghwan

        child1, child2 = do_uniform_mutation(parent1, parent2, lb, (1 / len(parent1.get_candidate_values())))
        population_to_return.append(child1)
        population_to_return.append(child2)

    return population_to_return


def correct(Q_T, lb, ub):
    for indx in range(len(Q_T)):
        candidate = Q_T[indx]
        values = candidate.get_candidate_values();
        for value_index in range(len(values)):
            if values[value_index] > ub[value_index] or values[value_index] < lb[value_index]:
                temp = generate_random_population(1, lb, ub)[0];
                Q_T[indx].set_candidate_values_at_index(value_index, temp.get_candidate_values()[value_index])

    return Q_T


def do_simulated_binary_crossover(parent1, parent2, nc=20):
    parent1 = parent1.get_candidate_values();
    parent2 = parent2.get_candidate_values()
    u = random.uniform(0, 1)
    # half Raja's code, as the child candidates was too close
    if u < 0.5:
        B = (2 * u) ** (1 / (nc + 1))
    else:
        B = (1 / (2 * (1 - u))) ** (1 / (nc + 1))
    t_parent1 = []
    t_parent2 = []

    for indx in range(len(parent1)):
        x1 = parent1[indx]
        x2 = parent2[indx]
        x1new = 0.5 * (((1 + B) * x1) + ((1 - B) * x2))
        x2new = 0.5 * (((1 - B) * x1) + ((1 + B) * x2))
        t_parent1.append(x1new)
        t_parent2.append(x2new)

    return Candidate(t_parent1), Candidate(t_parent2)


def do_gaussain_mutation_for_one(parent1_cand, lb, ub, thresh):
    parent1 = parent1_cand.get_candidate_values()

    for attrib in range(len(parent1)):
        if random.uniform(0, 1) > thresh:
            continue
        mu = 0;
        if attrib == 1:
            sigma = 1
        else:
            sigma = 10
        alpha = np.random.normal(mu, sigma)
        actualValueP1 = parent1[attrib];

        if (alpha < 1) and (alpha >= 0):
            if actualValueP1 + 1 < ub[attrib]:
                parent1[attrib] = parent1[attrib] + 1;


        elif (alpha <= 0) and (alpha > -1):
            if actualValueP1 - 1 > lb[attrib]:
                parent1[attrib] = parent1[attrib] - 1;
        else:
            if actualValueP1 + alpha < ub[attrib]:
                parent1[attrib] = parent1[attrib] + alpha;
    return Candidate(parent1)


def do_gaussain_mutation(parent1_cand, parent2_cand, lb, ub, thresh):
    parent1 = parent1_cand.get_candidate_values()
    parent2 = parent2_cand.get_candidate_values()
    for attrib in range(len(parent1)):
        random_value_for_theshold = random.uniform(0, 1);
        if random_value_for_theshold > thresh:
            continue
        mu = 0;
        if attrib == 1:
            sigma = 1
        else:
            sigma = 10
        alpha = np.random.normal(mu, sigma)
        actualValueP1 = parent1[attrib];
        actualValueP2 = parent2[attrib];

        if (alpha < 1) and (alpha >= 0):
            if actualValueP1 + 1 < ub[attrib]:
                parent1[attrib] = parent1[attrib] + 1;
            if actualValueP2 + 1 < ub[attrib]:
                parent2[attrib] = parent2[attrib] + 1;

        elif (alpha <= 0) and (alpha > -1):
            if actualValueP1 - 1 > lb[attrib]:
                parent1[attrib] = parent1[attrib] - 1;
            if actualValueP2 - 1 > lb[attrib]:
                parent2[attrib] = parent2[attrib] - 1;
        else:
            if actualValueP1 + alpha < ub[attrib]:
                parent1[attrib] = parent1[attrib] + alpha;
            if actualValueP2 + alpha < ub[attrib]:
                parent2[attrib] = parent2[attrib] + alpha;

    return Candidate(parent1), Candidate(parent2)


def get_distribution_index(parent1, parent2, objective_uncovered, threshold_criteria):
    total = 0;
    for each_obj in objective_uncovered:
        total = total + parent1.get_objective_value(each_obj)-0.95
        total = total + parent2.get_objective_value(each_obj)-0.95


    total=total/(len(objective_uncovered)*2)

    return 21-(total*400)



def recombine_improved(pop, objective_uncovered, lb, ub, threshold_criteria):
    size = len(pop)

    population_to_return = []

    if size == 1:
        candidate = do_gaussain_mutation_for_one(pop[0], lb, ub, (1 / len(pop[0].get_candidate_values())))
        population_to_return.append(candidate)

    else:
        while len(population_to_return) < size:
            parent1 = tournament_selection_improved(pop, 2, objective_uncovered)  # tournament selection same size as paper
            parent2 = tournament_selection_improved(pop, 2, objective_uncovered)
            while parent1 == parent2:
                parent2 = tournament_selection_improved(pop, 2, objective_uncovered)
            probability_crossover = random.uniform(0, 1)
            if probability_crossover <= 0.60:  # 60% probability
                print("getting distribution index")
                nc = get_distribution_index(parent1, parent2, objective_uncovered, threshold_criteria);
                parent1, parent2 = do_simulated_binary_crossover(parent1, parent2, nc)  # check with donghwan
            child1, child2 = do_gaussain_mutation(parent1, parent2, lb, ub, (1 / len(parent1.get_candidate_values())))

            population_to_return.append(child1)
            population_to_return.append(child2)

    return population_to_return


def recombine(pop, objective_uncovered, lb, ub):
    size = len(pop)

    population_to_return = []

    if size == 1:
        candidate = do_gaussain_mutation_for_one(pop[0], lb, ub, (1 / len(pop[0].get_candidate_values())))
        population_to_return.append(candidate)

    else:
        while len(population_to_return) < size:
            parent1 = tournament_selection(pop, 2, objective_uncovered)  # tournament selection same size as paper
            parent2 = tournament_selection(pop, 2, objective_uncovered)
            while parent1 == parent2:
                parent2 = tournament_selection(pop, 2, objective_uncovered)
            probability_crossover = random.uniform(0, 1)
            if probability_crossover <= 0.60:  # 60% probability
                parent1, parent2 = do_simulated_binary_crossover(parent1, parent2)  # check with donghwan
            child1, child2 = do_gaussain_mutation(parent1, parent2, lb, ub, (1 / len(parent1.get_candidate_values())))

            population_to_return.append(child1)
            population_to_return.append(child2)

    return population_to_return
