import numpy as np
from joblib.parallel import Parallel, delayed
from rich.progress import Progress


def evaluate_population(X, y, population):
    for individual in population:
        individual.optimize_leaves(X, y)
        individual.fitness = individual.evaluate(X, y)
    return population


def evolution_strategy_step(X, y, lamb, mu, population, best_fitness, best_tree, n_jobs=1):
    has_improved = False

    # Selecting μ parents
    population.sort(key=lambda x: x.fitness, reverse=True)
    parents = population[:mu]

    # Creating λ children
    children = []
    for parent in parents:
        for _ in range(lamb // mu):
            child = parent.copy()
            child.mutate()
            children.append(child)

    # Evaluating λ children
    children = Parallel(n_jobs=n_jobs)(delayed(evaluate_population)(X, y, children[i::n_jobs]) for i in range(n_jobs))
    children = [ind for sublist in children for ind in sublist]

    # Creating new population with μ parents and λ children
    population = parents + children

    # Saving best individual
    for individual in population:
        if individual.fitness > best_fitness:
            best_fitness = individual.fitness
            best_tree = individual
            has_improved = True

    return population, best_fitness, best_tree, has_improved


def evolution_strategy(config, tree_model, params, X, y, lamb, mu, n_gens, depth,
                       simulation_id=0, max_gens_wout_improvement=100, n_jobs=1, verbose=False):

    # Setup
    last_improvement_gen_id = 0
    best_fitness = -np.inf
    best_tree = None
    full_log = []

    # Initializing population with random models
    population = [tree_model.generate_random(config, depth, params, X) for _ in range(lamb)]
    population_par = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_population)(X, y, population[i::n_jobs]) for i in range(n_jobs))
    population = [ind for sublist in population_par for ind in sublist]

    for curr_gen in range(n_gens):
        # Evolution strategy step
        info = evolution_strategy_step(X, y, lamb, mu, population, best_fitness, best_tree, n_jobs=n_jobs)
        (population, best_fitness, best_tree, has_improved) = info

        # Logging
        fitnesses = [ind.fitness for ind in population]
        full_log.append((fitnesses, best_fitness, best_tree))

        if verbose:
            print(f"Simul. {simulation_id} // Generation {curr_gen} (last improv. {last_improvement_gen_id}) // Best fitness: {best_fitness}")

        # Early stopping
        if has_improved:
            last_improvement_gen_id = curr_gen
            if max_gens_wout_improvement is None or (curr_gen - last_improvement_gen_id) > max_gens_wout_improvement:
                break

    return best_tree, full_log


def evolution_strategy_tracked(config, tree_model, params, X, y, lamb, mu, n_gens, depth,
                               simulation_id=0, max_gens_wout_improvement=100, n_jobs=1):

    population = [tree_model.generate_random(config, depth, params, X) for _ in range(lamb)]
    population_par = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_population)(X, y, population[i::n_jobs]) for i in range(n_jobs))
    population = [ind for sublist in population_par for ind in sublist]

    last_improvement_gen_id = 0
    best_fitness = -np.inf
    best_tree = None
    full_log = []

    with Progress() as progress:
        task = progress.add_task("[bold green]Running ES...", total=n_gens)

        for curr_gen in range(n_gens):
            info = evolution_strategy_step(X, y, lamb, mu, population, best_fitness, best_tree, n_jobs=n_jobs)
            (population, best_fitness, best_tree, has_improved) = info

            fitnesses = [ind.fitness for ind in population]
            full_log.append((fitnesses, best_fitness, best_tree))

            if has_improved:
                last_improvement_gen_id = curr_gen

                if max_gens_wout_improvement is None or (curr_gen - last_improvement_gen_id) > max_gens_wout_improvement:
                    break

            progress.update(task, advance=1, description=f"[red]Running ES...[/red] "
                                                         f"[yellow]Simulation {simulation_id}[/yellow]"
                                                         f"[bright_black] // [/bright_black]"
                                                         f"[yellow]Generation {curr_gen}[/yellow] "
                                                         f"[cyan](last improv. {last_improvement_gen_id})[/cyan]"
                                                         f"[bright_black] // [/bright_black]"
                                                         f"[green]Best fitness:[/green] {'{:.5f}'.format(best_fitness)}")

    return best_tree, full_log
