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

    population_par = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_population)(X, y, population[i::n_jobs]) for i in range(n_jobs))
    population = [ind for sublist in population_par for ind in sublist]

    for individual in population:
        if individual.fitness > best_fitness:
            best_fitness = individual.fitness
            best_tree = individual
            has_improved = True

    population.sort(key=lambda x: x.fitness, reverse=True)
    parents = population[:mu]
    child_population = []

    for parent in parents:
        for _ in range(lamb // mu):
            child = parent.copy()
            child.mutate()
            child_population.append(child)

    population = parents + child_population

    return population, best_fitness, best_tree, has_improved


def evolution_strategy(config, tree_model, params, X, y, lamb, mu, n_gens, depth,
                       simulation_id=0, max_gens_wout_improvement=100, n_jobs=1, verbose=False):

    population = [tree_model.generate_random(config, depth, params, X) for _ in range(lamb)]
    for individual in population:
        individual.optimize_leaves(X, y)
        individual.fitness = individual.evaluate(X, y)

    last_improvement_gen_id = 0
    best_fitness = -np.inf
    best_tree = None
    full_log = []

    for curr_gen in range(n_gens):
        info = evolution_strategy_step(X, y, lamb, mu, population, best_fitness, best_tree, n_jobs=n_jobs)
        (population, best_fitness, best_tree, has_improved) = info

        fitnesses = [ind.fitness for ind in population]
        full_log.append((fitnesses, best_fitness, best_tree))

        if has_improved:
            last_improvement_gen_id = curr_gen

            if max_gens_wout_improvement is None or (curr_gen - last_improvement_gen_id) > max_gens_wout_improvement:
                print(f"Stopping early at gen #{curr_gen} (no improv. for {max_gens_wout_improvement} generations.")
                break

        if verbose:
            print(f"Simulation {simulation_id} // Generation {curr_gen} (last improv. {last_improvement_gen_id}) "
                  f"// Best fitness: {best_fitness}")

    return best_tree, full_log


def evolution_strategy_tracked(config, tree_model, params, X, y, lamb, mu, n_gens, depth,
                               simulation_id=0, max_gens_wout_improvement=100, n_jobs=1):

    population = [tree_model.generate_random(config, depth, params, X) for _ in range(lamb)]
    for individual in population:
        individual.optimize_leaves(X, y)
        individual.fitness = individual.evaluate(X, y)

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
                    print(f"Stopping early at gen #{curr_gen} (no improv. for {max_gens_wout_improvement} generations.")
                    break

            progress.update(task, advance=1, description=f"[red]Running ES...[/red] "
                                                         f"[yellow]Simulation {simulation_id}[/yellow]"
                                                         f"[bright_black] // [/bright_black]"
                                                         f"[yellow]Generation {curr_gen}[/yellow] "
                                                         f"[cyan](last improv. {last_improvement_gen_id})[/cyan]"
                                                         f"[bright_black] // [/bright_black]"
                                                         f"[green]Best fitness:[/green] {'{:.5f}'.format(best_fitness)}")

    return best_tree, full_log
