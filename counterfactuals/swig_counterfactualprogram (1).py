# # from aspmc.programs.problogprogram import ProblogProgram

# # """
# # Program module providing the algebraic progam class.
# # """
# # import time
# # import logging


# # import tempfile
# # import os 

# # import networkx as nx

# # import numpy as np
# # from ctypes import *
# # from array import array

# # from aspmc.programs.program import Rule

# # import aspmc.graph.treedecomposition as treedecomposition
# # from aspmc.compile.vtree import TD_to_vtree, TD_vtree
# # from aspmc.compile.dtree import TD_dtree
# # from pysdd.sdd import SddManager, Vtree, WmcManager
# # from aspmc.compile.cnf import CNF
# # from aspmc.compile.circuit import Circuit

# # from aspmc.config import config
# # from aspmc.util import *
# # from aspmc.programs.naming import *


# # import aspmc.signal_handling as my_signals

# # logger = logging.getLogger("WhatIf")

# # class SDDOperation(object):
# #     AND = 0
# #     OR = 1
# #     NEGATE = 2

# # class CounterfactualProgram(ProblogProgram):
# #     """A class for probabilistic programs that enables counterfactual inference. 

# #     This implementation uses a Single-World Intervention Graph (SWIG) approach.
# #     For each query, it transforms the program according to the specified intervention
# #     before incorporating evidence and computing the counterfactual probability.

# #     Args:
# #         program_str (:obj:`string`): A string containing a part of the program in ProbLog syntax. 
# #         May be the empty string.
# #         program_files (:obj:`list`): A list of string that are paths to files which contain programs in 
# #         ProbLog syntax that should be included. May be an empty list.

# #     Attributes:
# #         weights (:obj:`dict`): The dictionary from atom names to their weight.
# #         queries (:obj:`list`): The list of atoms to be queries in their string representation.
# #     """
# #     def __init__(self, program_str, program_files):
# #         # initialize the superclass
# #         ProblogProgram.__init__(self, program_str, program_files)
# #         if len(self.queries) > 0:
# #             logger.warning("Queries should not be included in the program specification. I will ignore them.")
# #             self.queries = []
        
# #         # The base class may store the program as a set. To add new rules and ensure 
# #         # a consistent type (list) for further processing, we convert it to a list, 
# #         # add our rule, and reassign. This mimics the structural pattern of the 
# #         # original implementation.
# #         new_program = list(self._program)
        
# #         # make sure there is always an atom `true` that is true
# #         self.true = self._new_var("true")
# #         self._deriv.add(self.true)
# #         new_program.append(Rule([self.true],[]))

# #         self._program = new_program


# #     def single_query(self, interventions, evidence, queries, strategy="sharpsat-td"):
# #         """Evaluates a single counterfactual query using the given strategy.

# #         This method implements the SWIG-based counterfactual reasoning approach
# #         as described in Algorithms 4 and 5 of the paper "Counterfactual Reasoning 
# #         in ProbLog via Single-World Intervention Graphs (SWIGS)". It first transforms 
# #         the program based on the intervention (SWIG-1) and then incorporates 
# #         evidence and runs marginal inference (SWIG-2).

# #         Args:
# #             interventions (dict): A dictionary mapping names to phases, 
# #                 indicating that the atom with name `name` should be intervened positively (phase == False) or negatively.
# #             evidence (dict): A dictionary mapping names to phases, 
# #                 indicating that the atom with name `name` must have been true (phase == False) or false.
# #             queries (list): A list of strings, indicating that we want to query the probabilities of the atoms
# #                 under the given interventions and evidence.
# #             strategy (:obj:`string`, optional): The knowledge compiler to use. Possible values are 
# #                 * `c2d`, `miniC2D`, `d4`, `sharpsat-td`.
# #                 Defaults to `sharpsat-td`.
# #         Returns:
# #             list: A list containing the results of the counterfactual queries in the order they were given in `queries`.
# #         """
# #         # The pysdd strategy was tied to the old twin-network implementation and is not supported.
# #         if strategy == 'pysdd':
# #             raise NotImplementedError("The pysdd strategy was part of the previous twin-network implementation and is not supported in the SWIG-based version.")

# #         # --- Algorithm 4: SWIG-1 Transformation ---
# #         tmp_program = []
        
# #         # Map from original intervened atom names to their new 'fixed' internal representation
# #         intervened_var_map = {}
# #         for name in interventions:
# #             original_var = self._name_to_var[name]
# #             # Create a new, unique variable for the fixed version of the intervened atom.
# #             fixed_var = self._new_var(f"{name}_fixed")
# #             intervened_var_map[original_var] = fixed_var

# #         # 1. Removal of Original Clauses & 2. Splitting of Nodes
# #         for rule in self._program:
# #             # Skip rules defining an intervened variable. 
# #             if len(rule.head) > 0 and rule.head[0] in intervened_var_map:
# #                 continue
            
# #             # In every other clause, replace occurrences of intervened variables in the body.
# #             new_body = []
# #             for literal in rule.body:
# #                 atom = abs(literal)
# #                 if atom in intervened_var_map:
# #                     # Replace with the fixed version of the atom. 
# #                     fixed_var = intervened_var_map[atom]
# #                     new_body.append(fixed_var if literal > 0 else -fixed_var)
# #                 else:
# #                     new_body.append(literal)
            
# #             tmp_program.append(Rule(rule.head, new_body))

# #         # 3. Adding Deterministic Facts for interventions. 
# #         for name, phase in interventions.items():
# #             original_var = self._name_to_var[name]
# #             fixed_var = intervened_var_map[original_var]
# #             # phase == False means intervene to be true; phase == True means intervene to be false.
# #             if not phase: 
# #                 # Add fact: 1.0::X_fixed.
# #                 tmp_program.append(Rule([fixed_var], []))
# #             else:
# #                 # Add constraint: :- X_fixed. (making it false).
# #                 tmp_program.append(Rule([], [fixed_var]))

# #         # --- Algorithm 5: Incorporating Evidence and Querying ---
        
# #         # Use original atom names for evidence and queries.
# #         atom_map = self._name_to_var

# #         # Incorporate Evidence by adding integrity constraints. 
# #         for name, phase in evidence.items():
# #             atom = atom_map[name]
# #             # if phase is True, evidence is that 'name' is false, so add rule :- name.
# #             # if phase is False, evidence is that 'name' is true, so add rule :- not name.
# #             if phase:
# #                 body = [atom]
# #             else:
# #                 body = [-atom]
# #             tmp_program.append(Rule([], body))

# #         # Query names refer to original atoms. Probability of evidence is calculated by querying 'true'.
# #         self.queries = ["true"] + queries
        
# #         program_string = self._prog_string(tmp_program)
# #         # Create a new probabilistic program for inference.
# #         inference_program = ProblogProgram(program_string, [])
# #         inference_program.queries = self.queries # Set queries for the inference engine.

# #         # Perform CNF conversion, followed by top down knowledge compilation.
# #         inference_program.td_guided_both_clark_completion(adaptive=False, latest=True)
# #         cnf = inference_program.get_cnf()
# #         result = cnf.evaluate(strategy="compilation", knowledge_compiler=strategy)
        
# #         # The result dictionary maps query names to their probabilities.
# #         prob_evidence = result.get("true", 0.0)

# #         if prob_evidence <= 0.0:
# #             raise Exception("Contradictory evidence! Probablity given evidence is zero.")

# #         # P(query | evidence) = P(query & evidence) / P(evidence)
# #         # The evaluation already computes P(query & evidence), so we just normalize. 
# #         final_results = [result.get(q, 0.0) / prob_evidence for q in queries]

# #         self.queries = []
# #         return final_results

# #     def multi_query(self, interventions, evidence, queries, strategy="sharpsat-td"):
# #         """Evaluates one of many single counterfactual queries using the given strategy.

# #         This method now serves as a wrapper for single_query to maintain a 
# #         consistent API. The previous implementation contained optimizations specific 
# #         to the twin-network approach, which are not applicable to the SWIG-based
# #         transformation model. Each call to multi_query will perform a new
# #         SWIG transformation and inference.

# #         Args:
# #             interventions (dict): A dictionary mapping names to phases, 
# #                 indicating that the atom with name `name` should be intervened positively (phase == False) or negatively.
# #             evidence (dict): A dictionary mapping names to phases, 
# #                 indicating that the atom with name `name` must have been true (phase == False) or false.
# #             queries (list): A list of strings, indicating that we want to query the probabilities of the atoms
# #                 under the given interventions and evidence.
# #             strategy (:obj:`string`, optional): The knowledge compiler to use. Possible values are 
# #                 * `c2d`, `miniC2D`, `d4`, `sharpsat-td`.
# #                 Defaults to `sharpsat-td`.
# #         Returns:
# #             list: A list containing the results of the counterfactual queries in the order they were given in `queries`.
# #         """
# #         if strategy == 'pysdd':
# #             raise NotImplementedError("The pysdd strategy was part of the previous twin-network implementation and is not supported in the SWIG-based version.")
        
# #         # The SWIG model requires re-transforming the program for each new set of
# #         # interventions. Therefore, we call single_query directly.
# #         return self.single_query(interventions, evidence, queries, strategy=strategy)

# from aspmc.programs.problogprogram import ProblogProgram

# """
# Program module providing the algebraic progam class.
# """
# import time
# import logging


# import tempfile
# import os 

# import networkx as nx

# import numpy as np
# from ctypes import *
# from array import array

# from aspmc.programs.program import Rule

# import aspmc.graph.treedecomposition as treedecomposition
# from aspmc.compile.vtree import TD_to_vtree, TD_vtree
# from aspmc.compile.dtree import TD_dtree
# from pysdd.sdd import SddManager, Vtree, WmcManager
# from aspmc.compile.cnf import CNF
# from aspmc.compile.circuit import Circuit

# from aspmc.config import config
# from aspmc.util import *
# from aspmc.programs.naming import *


# import aspmc.signal_handling as my_signals

# logger = logging.getLogger("WhatIf")

# class SDDOperation(object):
#     AND = 0
#     OR = 1
#     NEGATE = 2

# class CounterfactualProgram(ProblogProgram):
#     """A class for probabilistic programs that enables counterfactual inference. 

#     Should be specified in ProbLog syntax, but allows for stratification negation.

#     Grounding of these programs (and subclasses thereof) should follow the following strategy:

#     * `_prepare_grounding(self, program)` should take the output of the parser 
#         (i.e. a list of rules and special objects) and process all the rules and special objects
#         transforming them either into other rules or into strings that can be given to the grounder.
#     * the output of `_prepare_grounding(self, program)` is transformed to one program string via

#             '\\n'.join([ str(r) for r in program ])
        
#         This string will be given to the grounder, which produces a clingo control object.
#     * `_process_grounding(self, clingo_control)` should take this clingo control object and process the
#         grounding in an appropriate way (and draw some information from it optionally about weights, special objects).
#         The resulting processed clingo_control object must only know about the 
#         rules that should be seen in the base program class.

#     Thus, subclasses can override `_prepare_grounding` and `_process_grounding` (and optionally call the superclass methods) 
#     to handle their extras. See aspmc.programs.meuprogram or aspmc.programs.smprogram for examples.

#     Args:
#         program_str (:obj:`string`): A string containing a part of the program in ProbLog syntax. 
#         May be the empty string.
#         program_files (:obj:`list`): A list of string that are paths to files which contain programs in 
#         ProbLog syntax that should be included. May be an empty list.

#     Attributes:
#         weights (:obj:`dict`): The dictionary from atom names to their weight.
#         queries (:obj:`list`): The list of atoms to be queries in their string representation.
#     """
#     def __init__(self, program_str, program_files):
#         # initialize the superclass
#         ProblogProgram.__init__(self, program_str, program_files)
#         if len(self.queries) > 0:
#             logger.warning("Queries should not be included in the program specification. I will ignore them.")
#             self.queries = []
        
#         # attributes for the bottom up multi-query case
#         self._sdd_manager = None
#         self._topological_ordering = None
#         self._applyCache = {}
#         # attributes for the top down multi-query case
#         self._nnf = None
#         self._intervention_conditioners = {}
#         self._vtree = None

#         # new atoms for the SWIG model
#         self.evidence_atoms = {}
#         self.intervention_atoms = {}

#         new_program = []
#         for rule in self._program:
#             new_program.append(rule)

#         # make sure there is always an atom true that is true
#         self.true = self._new_var("true")
#         self._deriv.add(self.true)
#         new_program.append(Rule([self.true],[]))

#         self._program = new_program


#     def single_query(self, interventions, evidence, queries, strategy="sharpsat-td"):
#         """Evaluates a single counterfactual query using the given strategy.

#         Args:
#             interventions (dict): A dictionary mapping names to phases, 
#                 indicating that the atom with name `name` should be intervened positively (phase == False) or negatively.
#             evidence (dict): A dictionary mapping names to phases, 
#                 indicating that the atom with name `name` must have been true (phase == False) or false.
#             queries (list): A list of strings, indicating that we want to query the probabilities of the atoms
#                 under the given interventions and evidence.
#             strategy (:obj:`string`, optional): The knowledge compiler to use. Possible values are 
#                 * `pysdd` for bottom up compilation to SDDs,
#                 * `c2d` for top down compilation to sd-DNNF with c2d,
#                 * `miniC2D` for top down compilation to sd-DNNF with miniC2D,
#                 * `d4` for top down compilation to sd-DNNF with d4,
#                 * `sharpsat-td` for top down compilation to sd-DNNF with sharpsat-td.
#                 Defaults to `sharpsat-td`.
#         Returns:
#             list: A list containing the results of the counterfactual queries in the order they were given in `queries`.
#         """
#         # SWIG-1: Transformation under intervention
#         tmp_program = []
#         self.intervention_atoms = {name: self._new_var(f"{name}_fixed") for name in interventions}
        
#         # 1. Removal of Original Clauses
#         for rule in self._program:
#             if len(rule.head) > 0 and self._external_name(rule.head[0]) in interventions:
#                 continue
            
#             # 2. Splitting of Nodes
#             new_body = []
#             for atom in rule.body:
#                 atom_name = self._external_name(abs(atom))
#                 if atom_name in interventions:
#                     if atom > 0:
#                         new_body.append(self.intervention_atoms[atom_name])
#                     else:
#                         new_body.append(-self.intervention_atoms[atom_name])
#                 else:
#                     new_body.append(atom)
#             tmp_program.append(Rule(rule.head, new_body))

#         # 3. Adding Deterministic Facts
#         for name, phase in interventions.items():
#             atom = self.intervention_atoms[name]
#             if not phase:
#                 tmp_program.append(Rule([atom], []))

#         # SWIG-2: Incorporating Evidence and Querying
#         # 1. Apply SWIG Transformation (already done)
        
#         # 2. Incorporate Evidence
#         self.evidence_atoms = {name: self._new_var(f"{name}_obs") for name in evidence}
#         for name, phase in evidence.items():
#             atom = self.evidence_atoms[name]
#             # Adding facts to represent observation
#             if not phase:
#                 tmp_program.append(Rule([], [-atom]))
#             else:
#                 tmp_program.append(Rule([], [atom]))
        
#         # 3. Marginal Inference
#         # evaluate the query using the given strategy
#         if strategy in ['c2d', 'miniC2D', 'd4', 'sharpsat-td']:
#             # reduce the program to the relevant part
#             # set up the and/or graph
#             graph = nx.DiGraph()
#             for r in tmp_program:
#                 for atom in r.head:
#                     graph.add_edge(r, atom)
#                 for atom in r.body:
#                     graph.add_edge(abs(atom), r)
                    
#             # reduce to relevant part by using only the ancestors of evidence and or queries
#             relevant = set()
#             for query in queries:
#                 relevant.add(self._var_from_name(query))
#                 relevant.update(nx.ancestors(graph, self._var_from_name(query)))
#             for atom in evidence:
#                 relevant.add(self._var_from_name(atom))
#                 relevant.update(nx.ancestors(graph, self._var_from_name(atom)))

#             tmp_program = [ r for r in tmp_program if r in relevant ]
#             tmp_program.append(Rule([self.true], []))

#             # finalize the program with the evidence and the queries
#             for name, phase in evidence.items():
#                 atom = self._var_from_name(name)
#                 if phase:
#                     body = [ atom ]
#                 else:
#                     body = [ -atom ]
#                 tmp_program.append(Rule([],body))

#             self.queries = [ "true" ]
#             self.queries += queries
#             program_string = self._prog_string(tmp_program)
#             # create a new probabilistic program for inference
#             inference_program = ProblogProgram(program_string, [])
#             # perform CNF conversion, followed by top down knowledge compilation
#             inference_program.td_guided_both_clark_completion(adaptive = False, latest = True)
#             cnf = inference_program.get_cnf()
#             result = cnf.evaluate(strategy = "compilation")
#             # reorder the query results
#             other_queries = inference_program.get_queries()
#             to_idx = { query : idx for idx, query in enumerate(other_queries) }
#             sorted_result = [ ]
#             for query in self.queries:
#                 sorted_result.append(result[to_idx[query]])
#             if sorted_result[0] <= 0.0:
#                 raise Exception("Contradictory evidence! Probablity given evidence is zero.")
#             final_results = [ value/sorted_result[0] for value in sorted_result[1:] ] 
#         elif strategy == 'pysdd':
#             # perform bottom up compilation using pysdd
#             # set up the sdd manager
#             sdd = self.setup_sdd_manager(tmp_program)
#             vars = list(sdd.vars)
#             guesses = list(self._guess)
#             vertex_to_sdd = { v : vars[i] for i,v in enumerate(guesses) }

#             # set up the and/or graph
#             graph = nx.DiGraph()
#             for r in tmp_program:
#                 for atom in r.head:
#                     graph.add_edge(r, atom)
#                 for atom in r.body:
#                     graph.add_edge(abs(atom), r)

#             # reduce to relevant part by using only the ancestors of evidence and or queries
#             relevant = set()
#             for query in queries:
#                 relevant.add(self._var_from_name(query))
#                 relevant.update(nx.ancestors(graph, self._var_from_name(query)))
#             for atom in evidence:
#                 relevant.add(self._var_from_name(atom))
#                 relevant.update(nx.ancestors(graph, self._var_from_name(atom)))

#             start = time.process_time()
#             # build the relevant sdds by traversing the graph in topological order
#             ts = nx.topological_sort(graph)
#             for cur in ts:
#                 if cur not in relevant:
#                     continue
#                 if isinstance(cur, Rule):
#                     new_sdd = sdd.true()
#                     for b in cur.body:
#                         if b < 0:
#                             vertex_to_sdd[b] = ~vertex_to_sdd[-b]
#                         new_sdd = new_sdd & vertex_to_sdd[b]
#                     vertex_to_sdd[cur] = new_sdd
#                 elif cur not in self._guess:
#                     ins = list(graph.in_edges(nbunch=cur))
#                     new_sdd = sdd.false()
#                     for r in ins:
#                         new_sdd = new_sdd | vertex_to_sdd[r[0]]
#                     vertex_to_sdd[cur] = new_sdd

#             logger.info(f"  Time spent building the basic SDDs: {time.process_time() - start}")

#             start = time.process_time()
#             # conjoin all the evidence atoms
#             conjoined_evidence = sdd.true()
#             for name, phase in evidence.items():
#                 if phase:
#                     conjoined_evidence = conjoined_evidence & ~vertex_to_sdd[self._var_from_name(name)]
#                 else:
#                     conjoined_evidence = conjoined_evidence & vertex_to_sdd[self._var_from_name(name)]

#             # get all the query sdds and conjoin them with the evidence
#             query_sdds = [ vertex_to_sdd[self._var_from_name(query)] for query in queries ]
#             query_sdds = [ query_sdd & conjoined_evidence for query_sdd in query_sdds ]

#             logger.info(f"  Time spent conjoining the basic SDDs: {time.process_time() - start}")

#             # compute the actual probabilities
#             # first the probability of the evidence
#             evidence_manager = WmcManager(conjoined_evidence, log_mode = False)
#             weights = [ 1.0 for _ in range(2*len(self._guess)) ]
#             varMap = { name : var for var, name in self._nameMap.items() }
#             rev_mapping = { guesses[i] : i + 1 for i in range(len(self._guess)) }
#             for name in self.weights:
#                 sdd_var = rev_mapping[varMap[name]]
#                 weights[len(self._guess) + sdd_var - 1] = self.weights[name]
#                 weights[len(self._guess) - sdd_var] = 1 - self.weights[name]
#             python_array = np.array(weights)
#             c_weights = array('d', python_array.astype('float'))
#             evidence_manager.set_literal_weights_from_array(c_weights)
#             evidence_weight = evidence_manager.propagate()
#             if evidence_weight <= 0.0:
#                 raise Exception("Contradictory evidence! Probablity given evidence is zero.")
            
#             # then the probabilities of the queries given the evidence
#             final_results = []
#             for query_sdd in query_sdds:
#                 query_manager = WmcManager(query_sdd, log_mode = False)
#                 query_manager.set_literal_weights_from_array(c_weights)
#                 query_weight = query_manager.propagate()
#                 final_results.append(query_weight/evidence_weight)

#         self.queries = []
#         return final_results

#     def _setup_multiquery_bottom_up(self):
#         graph = nx.DiGraph()
#         for r in self._program:
#             for atom in r.head:
#                 graph.add_edge(r, atom)
#             for atom in r.body:
#                 graph.add_edge(abs(atom), r)
        
#         self._topological_ordering = list(nx.topological_sort(graph))
#         self._sdd_manager = self.setup_sdd_manager(self._program)

#         # perform bottom up compilation using pysdd
#         vars = list(self._sdd_manager.vars)
#         guesses = list(self._guess)
#         vertex_to_sdd = { v : vars[i] for i,v in enumerate(guesses) }


#         # build the relevant sdds by traversing the graph in topological order
#         # for better reuse we always take the same topological order 
#         # however, we need to make sure that we only have things in there that are relevant
#         # additionally, we now have new rules for the atoms that were intervened on
#         start = time.process_time()
#         ts = self._topological_ordering
#         for cur in ts:
#             if isinstance(cur, Rule):
#                 new_sdd = self._sdd_manager.true()
#                 for b in cur.body:
#                     if b < 0:
#                         vertex_to_sdd[b] = self._cached_apply(vertex_to_sdd[-b], None, SDDOperation.NEGATE)
#                     new_sdd = self._cached_apply(new_sdd, vertex_to_sdd[b], SDDOperation.AND)
#                 vertex_to_sdd[cur] = new_sdd
#             elif cur not in self._guess:
#                 ins = list(graph.in_edges(nbunch=cur))
#                 new_sdd = self._sdd_manager.false()
#                 for r in ins:
#                     new_sdd = self._cached_apply(new_sdd, vertex_to_sdd[r[0]], SDDOperation.OR)
#                 vertex_to_sdd[cur] = new_sdd

#         for atom_name in self.evidence_atoms:
#             atom = self._var_from_name(atom_name)
#             self._cached_apply(vertex_to_sdd[atom], None, SDDOperation.NEGATE)

#         logger.info(f"  Time spent building the basic SDDs in setup: {time.process_time() - start}")


#     def _setup_multiquery_top_down(self, strategy = "sharpsat-td"):
#         # create the atoms to condition on for interventions
#         # In SWIG, we handle interventions by modifying the program structure directly
#         # rather than adding special conditioning atoms.

#         self.td_guided_both_clark_completion(adaptive=False, latest=True)
#         cnf_fd, cnf_tmp = tempfile.mkstemp()
#         my_signals.tempfiles.add(cnf_tmp)
        
#         # prepare everything for the compilation
#         if strategy == "c2d":
#             with os.fdopen(cnf_fd, 'wb') as cnf_file:
#                 self._cnf.to_stream(cnf_file)
#             d3 = TD_dtree(self._cnf, solver = config["decos"], timeout = config["decot"])
#             d3.write(cnf_tmp + '.dtree')
#             my_signals.tempfiles.add(cnf_tmp + '.dtree')
#         elif strategy == "miniC2D":            
#             with os.fdopen(cnf_fd, 'wb') as cnf_file:
#                 self._cnf.to_stream(cnf_file)
#             self._vtree = TD_vtree(self._cnf, solver = config["decos"], timeout = config["decot"])
#             self._vtree.write(cnf_tmp + ".vtree")
#             my_signals.tempfiles.add(cnf_tmp + '.vtree')
#         elif strategy == "sharpsat-td":
#             with os.fdopen(cnf_fd, 'wb') as cnf_file:
#                 self._cnf.write_kc_cnf(cnf_file)
#         elif strategy == "d4":
#             with os.fdopen(cnf_fd, 'wb') as cnf_file:
#                 self._cnf.to_stream(cnf_file)
                
#         # perform the actual compilation
#         CNF.compile_single(cnf_tmp, knowledge_compiler = strategy)
        
#         # remove the temporary files
#         os.remove(cnf_tmp)
#         my_signals.tempfiles.remove(cnf_tmp)
#         self._nnf = cnf_tmp + ".nnf"
#         if strategy == "c2d":
#             os.remove(cnf_tmp + ".dtree")
#             my_signals.tempfiles.remove(cnf_tmp + '.dtree')
#         elif strategy == "miniC2D":
#             os.remove(cnf_tmp + ".vtree")
#             my_signals.tempfiles.remove(cnf_tmp + '.vtree')
        
#     def _cached_apply(self, node1, node2, operation):
#         if not (node1, node2, operation) in self._applyCache:
#             if operation == SDDOperation.AND:
#                 self._applyCache[(node1, node2, operation)] = node1 & node2
#             elif operation == SDDOperation.OR:
#                 self._applyCache[(node1, node2, operation)] = node1 | node2
#             elif operation == SDDOperation.NEGATE:
#                 assert(node2 is None)
#                 self._applyCache[(node1, node2, operation)] = ~node1
#         return self._applyCache[(node1, node2, operation)]

#     def multi_query(self, interventions, evidence, queries, strategy="sharpsat-td"):
#         """Evaluates one of many single counterfactual queries using the given strategy.

#         Args:
#             interventions (dict): A dictionary mapping names to phases, 
#                 indicating that the atom with name `name` should be intervened positively (phase == False) or negatively.
#             evidence (dict): A dictionary mapping names to phases, 
#                 indicating that the atom with name `name` must have been true (phase == False) or false.
#             queries (list): A list of strings, indicating that we want to query the probabilities of the atoms
#                 under the given interventions and evidence.
#             strategy (:obj:`string`, optional): The knowledge compiler to use. Possible values are 
#                 * `pysdd` for bottom up compilation to SDDs,
#                 * `c2d` for top down compilation to sd-DNNF with c2d,
#                 * `miniC2D` for top down compilation to sd-DNNF with miniC2D,
#                 * `d4` for top down compilation to sd-DNNF with d4,
#                 * `sharpsat-td` for top down compilation to sd-DNNF with sharpsat-td.
#                 Defaults to `sharpsat-td`.
#         Returns:
#             list: A list containing the results of the counterfactual queries in the order they were given in `queries`.
#         """
#         if strategy in ['c2d', 'miniC2D', 'd4', 'sharpsat-td']:
#             return self._multi_query_top_down(interventions, evidence, queries, strategy=strategy)
#         elif strategy == "pysdd":
#             return self._multi_query_bottom_up(interventions, evidence, queries, strategy=strategy)
#         else:
#             raise Exception(f"Unknown compilation strategy {strategy}.")

#     def _multi_query_top_down(self, interventions, evidence, queries, strategy="sharpsat-td"):
#         """Evaluates one of many single counterfactual queries using the given strategy.

#         Args:
#             interventions (dict): A dictionary mapping names to phases, 
#                 indicating that the atom with name `name` should be intervened positively (phase == False) or negatively.
#             evidence (dict): A dictionary mapping names to phases, 
#                 indicating that the atom with name `name` must have been true (phase == False) or false.
#             queries (list): A list of strings, indicating that we want to query the probabilities of the atoms
#                 under the given interventions and evidence.
#             strategy (:obj:`string`, optional): The knowledge compiler to use. Possible values are 
#                 * `c2d` for top down compilation to sd-DNNF with c2d,
#                 * `miniC2D` for top down compilation to sd-DNNF with miniC2D,
#                 * `d4` for top down compilation to sd-DNNF with d4,
#                 * `sharpsat-td` for top down compilation to sd-DNNF with sharpsat-td.
#                 Defaults to `sharpsat-td`.
#         Returns:
#             list: A list containing the results of the counterfactual queries in the order they were given in `queries`.
#         """
#         if self._nnf is None:
#             self._setup_multiquery_top_down(strategy=strategy)

#         # SWIG-1: Transformation for this query
#         prog = self._program[:]
#         intervention_atoms = {name: self._new_var(f"{name}_fixed") for name in interventions}

#         # 1. Removal of Original Clauses
#         rules_to_remove = []
#         for rule in prog:
#             if len(rule.head) > 0 and self._external_name(rule.head[0]) in interventions:
#                 rules_to_remove.append(rule)
#         for rule in rules_to_remove:
#             prog.remove(rule)

#         # 2. Splitting of Nodes and 3. Adding Deterministic Facts
#         for name, phase in interventions.items():
#             atom = intervention_atoms[name]
#             if not phase:
#                 prog.append(Rule([atom], []))
        
#         # SWIG-2: Incorporate evidence
#         for name, phase in evidence.items():
#             atom = self._var_from_name(name)
#             if phase:
#                 prog.append(Rule([], [atom]))
#             else:
#                 prog.append(Rule([], [-atom]))


#         # Temporarily set the program for inference
#         original_program = self._program
#         self._program = prog
        
#         # prepare the weights for this query
#         actual_queries = [ "true" ] + queries
#         query_cnt = len(actual_queries)
#         varMap = { name : var for var, name in self._nameMap.items() }
#         weight_list = [ np.full(query_cnt, self.semiring.one(), dtype=self.semiring.dtype) for _ in range(self._max*2) ]
#         for name in self.weights:
#             if name in varMap:
#                 weight_list[to_pos(varMap[name])] = np.full(query_cnt, self.weights[name], dtype=self.semiring.dtype)
#                 weight_list[neg(to_pos(varMap[name]))] = np.full(query_cnt, self.semiring.negate(self.weights[name]), dtype=self.semiring.dtype)
#         for i, query in enumerate(actual_queries):
#             if query in varMap:
#                 weight_list[neg(to_pos(varMap[query]))][i] = self.semiring.zero()
        
#         for v in range(self._max*2):
#             self._cnf.weights[to_dimacs(v)] = weight_list[v]
#         self._cnf.semirings = [ self.semiring ]
#         self._cnf.quantified = [ list(range(1, self._max + 1)) ]

#         # perform the counting on the circuit
#         weights, zero, one, dtype = self._cnf.get_weights()
#         results = Circuit.parse_wmc(self._nnf, weights, zero = zero, one = one, dtype = dtype, solver = strategy, vtree = self._vtree)
        
#         # Restore original program
#         self._program = original_program

#         if results[0] <= 0.0:
#             raise Exception("Contradictory evidence! Probablity given evidence is zero.")
        
#         final_results = [ result/results[0] for result in results[1:] ]
#         return final_results
        
#     def _multi_query_bottom_up(self, interventions, evidence, queries, strategy="pysdd"):
#         """Evaluates one of many single counterfactual queries using the given strategy.

#         Args:
#             interventions (dict): A dictionary mapping names to phases, 
#                 indicating that the atom with name `name` should be intervened positively (phase == False) or negatively.
#             evidence (dict): A dictionary mapping names to phases, 
#                 indicating that the atom with name `name` must have been true (phase == False) or false.
#             queries (list): A list of strings, indicating that we want to query the probabilities of the atoms
#                 under the given interventions and evidence.
#             strategy (:obj:`string`, optional): The knowledge compiler to use. Possible values are 
#                 * `pysdd` for bottom up compilation to SDDs,
#                 Defaults to `pysdd`.
#         Returns:
#             list: A list containing the results of the counterfactual queries in the order they were given in `queries`.
#         """
#         # check if setup already happened, if not do it now
#         if self._sdd_manager is None:
#             self._setup_multiquery_bottom_up()

#         # SWIG-1: Transformation
#         tmp_program = []
#         intervention_atoms = {name: self._new_var(f"{name}_fixed") for name in interventions}

#         # 1. Removal of Original Clauses
#         for rule in self._program:
#             if len(rule.head) > 0 and self._external_name(rule.head[0]) in interventions:
#                 continue
            
#             # 2. Splitting of Nodes
#             new_body = []
#             for atom in rule.body:
#                 atom_name = self._external_name(abs(atom))
#                 if atom_name in interventions:
#                     if atom > 0:
#                         new_body.append(intervention_atoms[atom_name])
#                     else:
#                         new_body.append(-intervention_atoms[atom_name])
#                 else:
#                     new_body.append(atom)
#             tmp_program.append(Rule(rule.head, new_body))

#         # 3. Adding Deterministic Facts
#         for name, phase in interventions.items():
#             if not phase:
#                 tmp_program.append(Rule([intervention_atoms[name]], []))

#         # perform bottom up compilation using pysdd
#         vars = list(self._sdd_manager.vars)
#         guesses = list(self._guess)
#         vertex_to_sdd = { v : vars[i] for i,v in enumerate(guesses) }

#         # set up the and/or graph
#         graph = nx.DiGraph()
#         for r in tmp_program:
#             for atom in r.head:
#                 graph.add_edge(r, atom)
#             for atom in r.body:
#                 graph.add_edge(abs(atom), r)
        
#         # reduce to relevant part by using only the ancestors of evidence and or queries
#         relevant = set()
#         for query in queries:
#             relevant.add(self._var_from_name(query))
#             relevant.update(nx.ancestors(graph, self._var_from_name(query)))
#         for name in evidence:
#             relevant.add(self._var_from_name(name))
#             relevant.update(nx.ancestors(graph, self._var_from_name(name)))

#         graph = nx.subgraph(graph, relevant)

#         # build the relevant sdds by traversing the graph in topological order
#         start = time.process_time()
#         ts = [ v for v in self._topological_ordering if v in relevant ]
#         for cur in ts:
#             if isinstance(cur, Rule):
#                 new_sdd = self._sdd_manager.true()
#                 for b in cur.body:
#                     if b < 0:
#                         vertex_to_sdd[b] = self._cached_apply(vertex_to_sdd[-b], None, SDDOperation.NEGATE)
#                     new_sdd = self._cached_apply(new_sdd, vertex_to_sdd[b], SDDOperation.AND)
#                 vertex_to_sdd[cur] = new_sdd
#             elif cur not in self._guess:
#                 ins = list(graph.in_edges(nbunch=cur))
#                 new_sdd = self._sdd_manager.false()
#                 for r in ins:
#                     new_sdd = self._cached_apply(new_sdd, vertex_to_sdd[r[0]], SDDOperation.OR)
#                 vertex_to_sdd[cur] = new_sdd

#         logger.info(f"  Time spent building the basic SDDs: {time.process_time() - start}")

#         start = time.process_time()
#         # conjoin all the evidence atoms
#         conjoined_evidence = self._sdd_manager.true()
#         for name, phase in evidence.items():
#             atom_var = self._var_from_name(name)
#             if phase:
#                 evidence_atom = self._cached_apply(vertex_to_sdd[atom_var], None, SDDOperation.NEGATE)
#             else:
#                 evidence_atom = vertex_to_sdd[atom_var]
#             conjoined_evidence = self._cached_apply(conjoined_evidence, evidence_atom, SDDOperation.AND)

#         # get all the query sdds and conjoin them with the evidence
#         query_sdds = [ vertex_to_sdd[self._var_from_name(query)] for query in queries ]
#         query_sdds = [ self._cached_apply(query_sdd, conjoined_evidence, SDDOperation.AND) for query_sdd in query_sdds ]

#         logger.info(f"  Time spent conjoining the basic SDDs: {time.process_time() - start}")
#         # compute the actual probabilities
#         # first the probability of the evidence
#         evidence_manager = WmcManager(conjoined_evidence, log_mode = False)
#         weights = [ 1.0 for _ in range(2*len(self._guess)) ]
#         varMap = { name : var for var, name in self._nameMap.items() }
#         rev_mapping = { guesses[i] : i + 1 for i in range(len(self._guess)) }
#         for name in self.weights:
#             if name in varMap:
#                 sdd_var = rev_mapping[varMap[name]]
#                 weights[len(self._guess) + sdd_var - 1] = self.weights[name]
#                 weights[len(self._guess) - sdd_var] = 1 - self.weights[name]
#         python_array = np.array(weights)
#         c_weights = array('d', python_array.astype('float'))
#         evidence_manager.set_literal_weights_from_array(c_weights)
#         evidence_weight = evidence_manager.propagate()
#         if evidence_weight <= 0.0:
#             raise Exception("Contradictory evidence! Probablity given evidence is zero.")
        
#         # then the probabilities of the queries given the evidence
#         final_results = []
#         for query_sdd in query_sdds:
#             query_manager = WmcManager(query_sdd, log_mode = False)
#             query_manager.set_literal_weights_from_array(c_weights)
#             query_weight = query_manager.propagate()
#             final_results.append(query_weight/evidence_weight)

#         return final_results
        
#     def _var_from_name(self, name):
#         for var, n in self._nameMap.items():
#             if n == name:
#                 return var
#         return None

#     def setup_sdd_manager(self, program):
#         # first generate a vtree for the program that is probably good
#         OR = 0
#         AND = 1
#         GUESS = 3
#         INPUT = 4
#         # approximate final width when using none strategy
#         nodes = { a : (OR, set()) for a in self._deriv }

#         cur_max = self._max
#         for a in self._exactlyOneOf:
#             cur_max += 1
#             nodes[cur_max] = (GUESS, set(abs(v) for v in a))

#         for atom in self._guess:
#             nodes[atom] = (INPUT, set())

#         for r in program:
#             cur_max += 1
#             nodes[cur_max] = (AND, set(abs(v) for v in r.body))
#             if len(r.head) != 0:
#                 nodes[abs(r.head[0])][1].add(cur_max)

#         # set up the and/or graph
#         graph = nx.Graph()
#         for a, inputs in nodes.items():
#             graph.add_edges_from([ (a, v) for v in inputs[1] ])
            
#         td = treedecomposition.from_graph(graph, solver = config["decos"], timeout = str(float(config["decot"])))
#         logger.info(f"Tree Decomposition #bags: {td.bags} unfolded treewidth: {td.width} #vertices: {td.vertices}")
#         td.remove(set(range(1, cur_max + 1)).difference(self._guess))
#         seen = set()
#         for bag in td.bag_iter():
#             seen.update(bag.vertices)
#         for unseen in self._guess.difference(seen):
#             td.bags += 1
#             bag = treedecomposition.Bag(td.bags, set([unseen]), [])
#             td.tree.add_edge(td.root, td.bags)
#             td.tree.nodes[td.bags]["bag"] = bag
#             td.get_root().children.append(bag)
#         my_vtree = TD_to_vtree(td)
#         guesses = list(self._guess)
#         rev_mapping = { guesses[i] : i + 1 for i in range(len(self._guess)) }
#         for node in my_vtree:
#             if node.val != None:
#                 assert(node.val in self._guess)
#                 node.val = rev_mapping[node.val]

#         (_, vtree_tmp) = tempfile.mkstemp()
#         my_vtree.write(vtree_tmp)
#         vtree = Vtree(filename=vtree_tmp)
#         os.remove(vtree_tmp)
#         sdd = SddManager.from_vtree(vtree)
        
#         return sdd


from aspmc.programs.problogprogram import ProblogProgram

"""
Program module providing the algebraic progam class.
"""
import time
import logging


import tempfile
import os 

import networkx as nx

import numpy as np
from ctypes import *
from array import array

from aspmc.programs.program import Rule

import aspmc.graph.treedecomposition as treedecomposition
from aspmc.compile.vtree import TD_to_vtree, TD_vtree
from aspmc.compile.dtree import TD_dtree
from pysdd.sdd import SddManager, Vtree, WmcManager
from aspmc.compile.cnf import CNF
from aspmc.compile.circuit import Circuit

from aspmc.config import config
from aspmc.util import *
from aspmc.programs.naming import *


import aspmc.signal_handling as my_signals

logger = logging.getLogger("WhatIf")

class SDDOperation(object):
    AND = 0
    OR = 1
    NEGATE = 2

class CounterfactualProgram(ProblogProgram):
    """A class for probabilistic programs that enables counterfactual inference. 

    Should be specified in ProbLog syntax, but allows for stratification negation.

    Grounding of these programs (and subclasses thereof) should follow the following strategy:

    * `_prepare_grounding(self, program)` should take the output of the parser 
        (i.e. a list of rules and special objects) and process all the rules and special objects
        transforming them either into other rules or into strings that can be given to the grounder.
    * the output of `_prepare_grounding(self, program)` is transformed to one program string via

            '\\n'.join([ str(r) for r in program ])
        
        This string will be given to the grounder, which produces a clingo control object.
    * `_process_grounding(self, clingo_control)` should take this clingo control object and process the
        grounding in an appropriate way (and draw some information from it optionally about weights, special objects).
        The resulting processed clingo_control object must only know about the 
        rules that should be seen in the base program class.

    Thus, subclasses can override `_prepare_grounding` and `_process_grounding` (and optionally call the superclass methods) 
    to handle their extras. See aspmc.programs.meuprogram or aspmc.programs.smprogram for examples.

    Args:
        program_str (:obj:`string`): A string containing a part of the program in ProbLog syntax. 
        May be the empty string.
        program_files (:obj:`list`): A list of string that are paths to files which contain programs in 
        ProbLog syntax that should be included. May be an empty list.

    Attributes:
        weights (:obj:`dict`): The dictionary from atom names to their weight.
        queries (:obj:`list`): The list of atoms to be queries in their string representation.
    """
    def __init__(self, program_str, program_files):
        # initialize the superclass
        ProblogProgram.__init__(self, program_str, program_files)
        if len(self.queries) > 0:
            logger.warning("Queries should not be included in the program specification. I will ignore them.")
            self.queries = []
        
        # attributes for the bottom up multi-query case
        self._sdd_manager = None
        self._topological_ordering = None
        self._applyCache = {}
        # attributes for the top down multi-query case
        self._nnf = None
        self._intervention_conditioners = {}
        self._vtree = None

        # new atoms for the SWIG model
        self.evidence_atoms = {}
        self.intervention_atoms = {}

        new_program = []
        for rule in self._program:
            new_program.append(rule)

        # make sure there is always an atom true that is true
        self.true = self._new_var("true")
        self._deriv.add(self.true)
        new_program.append(Rule([self.true],[]))

        self._program = new_program


    def single_query(self, interventions, evidence, queries, strategy="sharpsat-td"):
        """Evaluates a single counterfactual query using the given strategy.

        Args:
            interventions (dict): A dictionary mapping names to phases, 
                indicating that the atom with name `name` should be intervened positively (phase == False) or negatively.
            evidence (dict): A dictionary mapping names to phases, 
                indicating that the atom with name `name` must have been true (phase == False) or false.
            queries (list): A list of strings, indicating that we want to query the probabilities of the atoms
                under the given interventions and evidence.
            strategy (:obj:`string`, optional): The knowledge compiler to use. Possible values are 
                * `pysdd` for bottom up compilation to SDDs,
                * `c2d` for top down compilation to sd-DNNF with c2d,
                * `miniC2D` for top down compilation to sd-DNNF with miniC2D,
                * `d4` for top down compilation to sd-DNNF with d4,
                * `sharpsat-td` for top down compilation to sd-DNNF with sharpsat-td.
                Defaults to `sharpsat-td`.
        Returns:
            list: A list containing the results of the counterfactual queries in the order they were given in `queries`.
        """
        # SWIG-1: Transformation under intervention
        tmp_program = []
        # self.intervention_atoms = {name: self._new_var(f"{name}_fixed") for name in interventions}
        self.intervention_atoms = {name: self._new_var(f"fixed({name})") for name in interventions}

        # 1. Removal of Original Clauses
        for rule in self._program:
            if len(rule.head) > 0 and self._external_name(rule.head[0]) in interventions:
                continue
            
            # 2. Splitting of Nodes
            new_body = []
            for atom in rule.body:
                atom_name = self._external_name(abs(atom))
                if atom_name in interventions:
                    if atom > 0:
                        new_body.append(self.intervention_atoms[atom_name])
                    else:
                        new_body.append(-self.intervention_atoms[atom_name])
                else:
                    new_body.append(atom)
            tmp_program.append(Rule(rule.head, new_body))

        # 3. Adding Deterministic Facts
        for name, phase in interventions.items():
            atom = self.intervention_atoms[name]
            if not phase:
                tmp_program.append(Rule([atom], []))

        # SWIG-2: Incorporating Evidence and Querying
        # 1. Apply SWIG Transformation (already done)
        
        # 2. Incorporate Evidence
        # self.evidence_atoms = {name: self._new_var(f"{name}_obs") for name in evidence}
        self.evidence_atoms = {name: self._new_var(f"obs({name})") for name in evidence}

        # for name, phase in evidence.items():
        #     atom = self.evidence_atoms[name]
        #     # Adding facts to represent observation
        #     if not phase:
        #         tmp_program.append(Rule([], [-atom]))
        #     else:
        #         tmp_program.append(Rule([], [atom]))
        for name, phase in evidence.items():
            atom = self.evidence_atoms[name]
            # Link the observed atom to the original atom: obs_atom :- original_atom.
            tmp_program.append(Rule([atom], [self._var_from_name(name)]))
            if not phase: # If evidence is TRUE
                # Add constraint: not obs_atom is false.
                tmp_program.append(Rule([], [-atom]))
            else: # If evidence is FALSE
                # Add constraint: obs_atom is false.
                tmp_program.append(Rule([], [atom]))
        # 3. Marginal Inference
        # evaluate the query using the given strategy
        if strategy in ['c2d', 'miniC2D', 'd4', 'sharpsat-td']:
            # reduce the program to the relevant part
            # set up the and/or graph
            graph = nx.DiGraph()
            for r in tmp_program:
                for atom in r.head:
                    graph.add_edge(r, atom)
                for atom in r.body:
                    graph.add_edge(abs(atom), r)
                    
            # reduce to relevant part by using only the ancestors of evidence and or queries
            relevant = set()
            for query in queries:
                relevant.add(self._var_from_name(query))
                relevant.update(nx.ancestors(graph, self._var_from_name(query)))
            for atom in evidence:
                relevant.add(self._var_from_name(atom))
                relevant.update(nx.ancestors(graph, self._var_from_name(atom)))

            tmp_program = [ r for r in tmp_program if r in relevant ]
            tmp_program.append(Rule([self.true], []))

            # finalize the program with the evidence and the queries
            for name, phase in evidence.items():
                atom = self._var_from_name(name)
                if phase:
                    body = [ atom ]
                else:
                    body = [ -atom ]
                tmp_program.append(Rule([],body))

            self.queries = [ "true" ]
            self.queries += queries
            program_string = self._prog_string(tmp_program)
            # create a new probabilistic program for inference
            inference_program = ProblogProgram(program_string, [])
            # perform CNF conversion, followed by top down knowledge compilation
            inference_program.td_guided_both_clark_completion(adaptive = False, latest = True)
            cnf = inference_program.get_cnf()
            result = cnf.evaluate(strategy = "compilation")
            # reorder the query results
            other_queries = inference_program.get_queries()
            to_idx = { query : idx for idx, query in enumerate(other_queries) }
            sorted_result = [ ]
            for query in self.queries:
                sorted_result.append(result[to_idx[query]])
            if sorted_result[0] <= 0.0:
                raise Exception("Contradictory evidence! Probablity given evidence is zero.")
            final_results = [ value/sorted_result[0] for value in sorted_result[1:] ] 
        elif strategy == 'pysdd':
            # perform bottom up compilation using pysdd
            # set up the sdd manager
            sdd = self.setup_sdd_manager(tmp_program)
            vars = list(sdd.vars)
            guesses = list(self._guess)
            vertex_to_sdd = { v : vars[i] for i,v in enumerate(guesses) }

            # set up the and/or graph
            graph = nx.DiGraph()
            for r in tmp_program:
                for atom in r.head:
                    graph.add_edge(r, atom)
                for atom in r.body:
                    graph.add_edge(abs(atom), r)

            # reduce to relevant part by using only the ancestors of evidence and or queries
            relevant = set()
            for query in queries:
                relevant.add(self._var_from_name(query))
                relevant.update(nx.ancestors(graph, self._var_from_name(query)))
            for atom in evidence:
                relevant.add(self._var_from_name(atom))
                relevant.update(nx.ancestors(graph, self._var_from_name(atom)))

            start = time.process_time()
            # build the relevant sdds by traversing the graph in topological order
            ts = nx.topological_sort(graph)
            for cur in ts:
                if cur not in relevant:
                    continue
                if isinstance(cur, Rule):
                    new_sdd = sdd.true()
                    for b in cur.body:
                        if b < 0:
                            vertex_to_sdd[b] = ~vertex_to_sdd[-b]
                        new_sdd = new_sdd & vertex_to_sdd[b]
                    vertex_to_sdd[cur] = new_sdd
                elif cur not in self._guess:
                    ins = list(graph.in_edges(nbunch=cur))
                    new_sdd = sdd.false()
                    for r in ins:
                        new_sdd = new_sdd | vertex_to_sdd[r[0]]
                    vertex_to_sdd[cur] = new_sdd

            logger.info(f"  Time spent building the basic SDDs: {time.process_time() - start}")

            start = time.process_time()
            # conjoin all the evidence atoms
            conjoined_evidence = sdd.true()
            for name, phase in evidence.items():
                if phase:
                    conjoined_evidence = conjoined_evidence & ~vertex_to_sdd[self._var_from_name(name)]
                else:
                    conjoined_evidence = conjoined_evidence & vertex_to_sdd[self._var_from_name(name)]

            # get all the query sdds and conjoin them with the evidence
            query_sdds = [ vertex_to_sdd[self._var_from_name(query)] for query in queries ]
            query_sdds = [ query_sdd & conjoined_evidence for query_sdd in query_sdds ]

            logger.info(f"  Time spent conjoining the basic SDDs: {time.process_time() - start}")

            # compute the actual probabilities
            # first the probability of the evidence
            evidence_manager = WmcManager(conjoined_evidence, log_mode = False)
            weights = [ 1.0 for _ in range(2*len(self._guess)) ]
            varMap = { name : var for var, name in self._nameMap.items() }
            rev_mapping = { guesses[i] : i + 1 for i in range(len(self._guess)) }
            for name in self.weights:
                sdd_var = rev_mapping[varMap[name]]
                weights[len(self._guess) + sdd_var - 1] = self.weights[name]
                weights[len(self._guess) - sdd_var] = 1 - self.weights[name]
            python_array = np.array(weights)
            c_weights = array('d', python_array.astype('float'))
            evidence_manager.set_literal_weights_from_array(c_weights)
            evidence_weight = evidence_manager.propagate()
            if evidence_weight <= 0.0:
                raise Exception("Contradictory evidence! Probablity given evidence is zero.")
            
            # then the probabilities of the queries given the evidence
            final_results = []
            for query_sdd in query_sdds:
                query_manager = WmcManager(query_sdd, log_mode = False)
                query_manager.set_literal_weights_from_array(c_weights)
                query_weight = query_manager.propagate()
                final_results.append(query_weight/evidence_weight)

        self.queries = []
        return final_results

    def _setup_multiquery_bottom_up(self):
        graph = nx.DiGraph()
        for r in self._program:
            for atom in r.head:
                graph.add_edge(r, atom)
            for atom in r.body:
                graph.add_edge(abs(atom), r)
        
        self._topological_ordering = list(nx.topological_sort(graph))
        self._sdd_manager = self.setup_sdd_manager(self._program)

        # perform bottom up compilation using pysdd
        vars = list(self._sdd_manager.vars)
        guesses = list(self._guess)
        vertex_to_sdd = { v : vars[i] for i,v in enumerate(guesses) }


        # build the relevant sdds by traversing the graph in topological order
        # for better reuse we always take the same topological order 
        # however, we need to make sure that we only have things in there that are relevant
        # additionally, we now have new rules for the atoms that were intervened on
        start = time.process_time()
        ts = self._topological_ordering
        for cur in ts:
            if isinstance(cur, Rule):
                new_sdd = self._sdd_manager.true()
                for b in cur.body:
                    if b < 0:
                        vertex_to_sdd[b] = self._cached_apply(vertex_to_sdd[-b], None, SDDOperation.NEGATE)
                    new_sdd = self._cached_apply(new_sdd, vertex_to_sdd[b], SDDOperation.AND)
                vertex_to_sdd[cur] = new_sdd
            elif cur not in self._guess:
                ins = list(graph.in_edges(nbunch=cur))
                new_sdd = self._sdd_manager.false()
                for r in ins:
                    new_sdd = self._cached_apply(new_sdd, vertex_to_sdd[r[0]], SDDOperation.OR)
                vertex_to_sdd[cur] = new_sdd

        for atom_name in self.evidence_atoms:
            atom = self._var_from_name(atom_name)
            self._cached_apply(vertex_to_sdd[atom], None, SDDOperation.NEGATE)

        logger.info(f"  Time spent building the basic SDDs in setup: {time.process_time() - start}")


    def _setup_multiquery_top_down(self, strategy = "sharpsat-td"):
        # create the atoms to condition on for interventions
        # In SWIG, we handle interventions by modifying the program structure directly
        # rather than adding special conditioning atoms.

        self.td_guided_both_clark_completion(adaptive=False, latest=True)
        cnf_fd, cnf_tmp = tempfile.mkstemp()
        my_signals.tempfiles.add(cnf_tmp)
        
        # prepare everything for the compilation
        if strategy == "c2d":
            with os.fdopen(cnf_fd, 'wb') as cnf_file:
                self._cnf.to_stream(cnf_file)
            d3 = TD_dtree(self._cnf, solver = config["decos"], timeout = config["decot"])
            d3.write(cnf_tmp + '.dtree')
            my_signals.tempfiles.add(cnf_tmp + '.dtree')
        elif strategy == "miniC2D":            
            with os.fdopen(cnf_fd, 'wb') as cnf_file:
                self._cnf.to_stream(cnf_file)
            self._vtree = TD_vtree(self._cnf, solver = config["decos"], timeout = config["decot"])
            self._vtree.write(cnf_tmp + ".vtree")
            my_signals.tempfiles.add(cnf_tmp + '.vtree')
        elif strategy == "sharpsat-td":
            with os.fdopen(cnf_fd, 'wb') as cnf_file:
                self._cnf.write_kc_cnf(cnf_file)
        elif strategy == "d4":
            with os.fdopen(cnf_fd, 'wb') as cnf_file:
                self._cnf.to_stream(cnf_file)
                
        # perform the actual compilation
        CNF.compile_single(cnf_tmp, knowledge_compiler = strategy)
        
        # remove the temporary files
        os.remove(cnf_tmp)
        my_signals.tempfiles.remove(cnf_tmp)
        self._nnf = cnf_tmp + ".nnf"
        if strategy == "c2d":
            os.remove(cnf_tmp + ".dtree")
            my_signals.tempfiles.remove(cnf_tmp + '.dtree')
        elif strategy == "miniC2D":
            os.remove(cnf_tmp + ".vtree")
            my_signals.tempfiles.remove(cnf_tmp + '.vtree')
        
    def _cached_apply(self, node1, node2, operation):
        if not (node1, node2, operation) in self._applyCache:
            if operation == SDDOperation.AND:
                self._applyCache[(node1, node2, operation)] = node1 & node2
            elif operation == SDDOperation.OR:
                self._applyCache[(node1, node2, operation)] = node1 | node2
            elif operation == SDDOperation.NEGATE:
                assert(node2 is None)
                self._applyCache[(node1, node2, operation)] = ~node1
        return self._applyCache[(node1, node2, operation)]

    def multi_query(self, interventions, evidence, queries, strategy="sharpsat-td"):
        """Evaluates one of many single counterfactual queries using the given strategy.

        Args:
            interventions (dict): A dictionary mapping names to phases, 
                indicating that the atom with name `name` should be intervened positively (phase == False) or negatively.
            evidence (dict): A dictionary mapping names to phases, 
                indicating that the atom with name `name` must have been true (phase == False) or false.
            queries (list): A list of strings, indicating that we want to query the probabilities of the atoms
                under the given interventions and evidence.
            strategy (:obj:`string`, optional): The knowledge compiler to use. Possible values are 
                * `pysdd` for bottom up compilation to SDDs,
                * `c2d` for top down compilation to sd-DNNF with c2d,
                * `miniC2D` for top down compilation to sd-DNNF with miniC2D,
                * `d4` for top down compilation to sd-DNNF with d4,
                * `sharpsat-td` for top down compilation to sd-DNNF with sharpsat-td.
                Defaults to `sharpsat-td`.
        Returns:
            list: A list containing the results of the counterfactual queries in the order they were given in `queries`.
        """
        if strategy in ['c2d', 'miniC2D', 'd4', 'sharpsat-td']:
            return self._multi_query_top_down(interventions, evidence, queries, strategy=strategy)
        elif strategy == "pysdd":
            return self._multi_query_bottom_up(interventions, evidence, queries, strategy=strategy)
        else:
            raise Exception(f"Unknown compilation strategy {strategy}.")

    def _multi_query_top_down(self, interventions, evidence, queries, strategy="sharpsat-td"):
        """Evaluates one of many single counterfactual queries using the given strategy.

        Args:
            interventions (dict): A dictionary mapping names to phases, 
                indicating that the atom with name `name` should be intervened positively (phase == False) or negatively.
            evidence (dict): A dictionary mapping names to phases, 
                indicating that the atom with name `name` must have been true (phase == False) or false.
            queries (list): A list of strings, indicating that we want to query the probabilities of the atoms
                under the given interventions and evidence.
            strategy (:obj:`string`, optional): The knowledge compiler to use. Possible values are 
                * `c2d` for top down compilation to sd-DNNF with c2d,
                * `miniC2D` for top down compilation to sd-DNNF with miniC2D,
                * `d4` for top down compilation to sd-DNNF with d4,
                * `sharpsat-td` for top down compilation to sd-DNNF with sharpsat-td.
                Defaults to `sharpsat-td`.
        Returns:
            list: A list containing the results of the counterfactual queries in the order they were given in `queries`.
        """
        if self._nnf is None:
            self._setup_multiquery_top_down(strategy=strategy)

        # SWIG-1: Transformation for this query
        prog = self._program[:]
        # intervention_atoms = {name: self._new_var(f"{name}_fixed") for name in interventions}
        intervention_atoms = {name: self._new_var(f"fixed({name})") for name in interventions}

        # 1. Removal of Original Clauses
        rules_to_remove = []
        for rule in prog:
            if len(rule.head) > 0 and self._external_name(rule.head[0]) in interventions:
                rules_to_remove.append(rule)
        for rule in rules_to_remove:
            prog.remove(rule)

        # 2. Splitting of Nodes and 3. Adding Deterministic Facts
        for name, phase in interventions.items():
            atom = intervention_atoms[name]
            if not phase:
                prog.append(Rule([atom], []))
        
        # SWIG-2: Incorporate evidence
        for name, phase in evidence.items():
            atom = self._var_from_name(name)
            if phase:
                prog.append(Rule([], [atom]))
            else:
                prog.append(Rule([], [-atom]))


        # Temporarily set the program for inference
        original_program = self._program
        self._program = prog
        
        # prepare the weights for this query
        actual_queries = [ "true" ] + queries
        query_cnt = len(actual_queries)
        varMap = { name : var for var, name in self._nameMap.items() }
        weight_list = [ np.full(query_cnt, self.semiring.one(), dtype=self.semiring.dtype) for _ in range(self._max*2) ]
        for name in self.weights:
            if name in varMap:
                weight_list[to_pos(varMap[name])] = np.full(query_cnt, self.weights[name], dtype=self.semiring.dtype)
                weight_list[neg(to_pos(varMap[name]))] = np.full(query_cnt, self.semiring.negate(self.weights[name]), dtype=self.semiring.dtype)
        for i, query in enumerate(actual_queries):
            if query in varMap:
                weight_list[neg(to_pos(varMap[query]))][i] = self.semiring.zero()
        
        for v in range(self._max*2):
            self._cnf.weights[to_dimacs(v)] = weight_list[v]
        self._cnf.semirings = [ self.semiring ]
        self._cnf.quantified = [ list(range(1, self._max + 1)) ]

        # perform the counting on the circuit
        weights, zero, one, dtype = self._cnf.get_weights()
        results = Circuit.parse_wmc(self._nnf, weights, zero = zero, one = one, dtype = dtype, solver = strategy, vtree = self._vtree)
        
        # Restore original program
        self._program = original_program

        if results[0] <= 0.0:
            raise Exception("Contradictory evidence! Probablity given evidence is zero.")
        
        final_results = [ result/results[0] for result in results[1:] ]
        return final_results
        
    def _multi_query_bottom_up(self, interventions, evidence, queries, strategy="pysdd"):
        """Evaluates one of many single counterfactual queries using the given strategy.

        Args:
            interventions (dict): A dictionary mapping names to phases, 
                indicating that the atom with name `name` should be intervened positively (phase == False) or negatively.
            evidence (dict): A dictionary mapping names to phases, 
                indicating that the atom with name `name` must have been true (phase == False) or false.
            queries (list): A list of strings, indicating that we want to query the probabilities of the atoms
                under the given interventions and evidence.
            strategy (:obj:`string`, optional): The knowledge compiler to use. Possible values are 
                * `pysdd` for bottom up compilation to SDDs,
                Defaults to `pysdd`.
        Returns:
            list: A list containing the results of the counterfactual queries in the order they were given in `queries`.
        """
        # check if setup already happened, if not do it now
        if self._sdd_manager is None:
            self._setup_multiquery_bottom_up()

        # SWIG-1: Transformation
        tmp_program = []
        # intervention_atoms = {name: self._new_var(f"{name}_fixed") for name in interventions}
        intervention_atoms = {name: self._new_var(f"fixed({name})") for name in interventions}

        # 1. Removal of Original Clauses
        for rule in self._program:
            if len(rule.head) > 0 and self._external_name(rule.head[0]) in interventions:
                continue
            
            # 2. Splitting of Nodes
            new_body = []
            for atom in rule.body:
                atom_name = self._external_name(abs(atom))
                if atom_name in interventions:
                    if atom > 0:
                        new_body.append(intervention_atoms[atom_name])
                    else:
                        new_body.append(-intervention_atoms[atom_name])
                else:
                    new_body.append(atom)
            tmp_program.append(Rule(rule.head, new_body))

        # 3. Adding Deterministic Facts
        for name, phase in interventions.items():
            if not phase:
                tmp_program.append(Rule([intervention_atoms[name]], []))

        # perform bottom up compilation using pysdd
        vars = list(self._sdd_manager.vars)
        guesses = list(self._guess)
        vertex_to_sdd = { v : vars[i] for i,v in enumerate(guesses) }

        # set up the and/or graph
        graph = nx.DiGraph()
        for r in tmp_program:
            for atom in r.head:
                graph.add_edge(r, atom)
            for atom in r.body:
                graph.add_edge(abs(atom), r)
        
        # reduce to relevant part by using only the ancestors of evidence and or queries
        relevant = set()
        for query in queries:
            relevant.add(self._var_from_name(query))
            relevant.update(nx.ancestors(graph, self._var_from_name(query)))
        for name in evidence:
            relevant.add(self._var_from_name(name))
            relevant.update(nx.ancestors(graph, self._var_from_name(name)))

        graph = nx.subgraph(graph, relevant)

        # build the relevant sdds by traversing the graph in topological order
        start = time.process_time()
        ts = [ v for v in self._topological_ordering if v in relevant ]
        for cur in ts:
            if isinstance(cur, Rule):
                new_sdd = self._sdd_manager.true()
                for b in cur.body:
                    if b < 0:
                        vertex_to_sdd[b] = self._cached_apply(vertex_to_sdd[-b], None, SDDOperation.NEGATE)
                    new_sdd = self._cached_apply(new_sdd, vertex_to_sdd[b], SDDOperation.AND)
                vertex_to_sdd[cur] = new_sdd
            elif cur not in self._guess:
                ins = list(graph.in_edges(nbunch=cur))
                new_sdd = self._sdd_manager.false()
                for r in ins:
                    new_sdd = self._cached_apply(new_sdd, vertex_to_sdd[r[0]], SDDOperation.OR)
                vertex_to_sdd[cur] = new_sdd

        logger.info(f"  Time spent building the basic SDDs: {time.process_time() - start}")

        start = time.process_time()
        # conjoin all the evidence atoms
        conjoined_evidence = self._sdd_manager.true()
        for name, phase in evidence.items():
            atom_var = self._var_from_name(name)
            if phase:
                evidence_atom = self._cached_apply(vertex_to_sdd[atom_var], None, SDDOperation.NEGATE)
            else:
                evidence_atom = vertex_to_sdd[atom_var]
            conjoined_evidence = self._cached_apply(conjoined_evidence, evidence_atom, SDDOperation.AND)

        # get all the query sdds and conjoin them with the evidence
        query_sdds = [ vertex_to_sdd[self._var_from_name(query)] for query in queries ]
        query_sdds = [ self._cached_apply(query_sdd, conjoined_evidence, SDDOperation.AND) for query_sdd in query_sdds ]

        logger.info(f"  Time spent conjoining the basic SDDs: {time.process_time() - start}")
        # compute the actual probabilities
        # first the probability of the evidence
        evidence_manager = WmcManager(conjoined_evidence, log_mode = False)
        weights = [ 1.0 for _ in range(2*len(self._guess)) ]
        varMap = { name : var for var, name in self._nameMap.items() }
        rev_mapping = { guesses[i] : i + 1 for i in range(len(self._guess)) }
        for name in self.weights:
            if name in varMap:
                sdd_var = rev_mapping[varMap[name]]
                weights[len(self._guess) + sdd_var - 1] = self.weights[name]
                weights[len(self._guess) - sdd_var] = 1 - self.weights[name]
        python_array = np.array(weights)
        c_weights = array('d', python_array.astype('float'))
        evidence_manager.set_literal_weights_from_array(c_weights)
        evidence_weight = evidence_manager.propagate()
        if evidence_weight <= 0.0:
            raise Exception("Contradictory evidence! Probablity given evidence is zero.")
        
        # then the probabilities of the queries given the evidence
        final_results = []
        for query_sdd in query_sdds:
            query_manager = WmcManager(query_sdd, log_mode = False)
            query_manager.set_literal_weights_from_array(c_weights)
            query_weight = query_manager.propagate()
            final_results.append(query_weight/evidence_weight)

        return final_results
        
    def _var_from_name(self, name):
        for var, n in self._nameMap.items():
            if n == name:
                return var
        return None

    def setup_sdd_manager(self, program):
        # first generate a vtree for the program that is probably good
        OR = 0
        AND = 1
        GUESS = 3
        INPUT = 4
        # approximate final width when using none strategy
        nodes = { a : (OR, set()) for a in self._deriv }

        cur_max = self._max
        for a in self._exactlyOneOf:
            cur_max += 1
            nodes[cur_max] = (GUESS, set(abs(v) for v in a))

        for atom in self._guess:
            nodes[atom] = (INPUT, set())

        for r in program:
            cur_max += 1
            nodes[cur_max] = (AND, set(abs(v) for v in r.body))
            if len(r.head) != 0:
                nodes[abs(r.head[0])][1].add(cur_max)

        # set up the and/or graph
        graph = nx.Graph()
        for a, inputs in nodes.items():
            graph.add_edges_from([ (a, v) for v in inputs[1] ])
            
        td = treedecomposition.from_graph(graph, solver = config["decos"], timeout = str(float(config["decot"])))
        logger.info(f"Tree Decomposition #bags: {td.bags} unfolded treewidth: {td.width} #vertices: {td.vertices}")
        td.remove(set(range(1, cur_max + 1)).difference(self._guess))
        seen = set()
        for bag in td.bag_iter():
            seen.update(bag.vertices)
        for unseen in self._guess.difference(seen):
            td.bags += 1
            bag = treedecomposition.Bag(td.bags, set([unseen]), [])
            td.tree.add_edge(td.root, td.bags)
            td.tree.nodes[td.bags]["bag"] = bag
            td.get_root().children.append(bag)
        my_vtree = TD_to_vtree(td)
        guesses = list(self._guess)
        rev_mapping = { guesses[i] : i + 1 for i in range(len(self._guess)) }
        for node in my_vtree:
            if node.val != None:
                assert(node.val in self._guess)
                node.val = rev_mapping[node.val]

        (_, vtree_tmp) = tempfile.mkstemp()
        my_vtree.write(vtree_tmp)
        vtree = Vtree(filename=vtree_tmp)
        os.remove(vtree_tmp)
        sdd = SddManager.from_vtree(vtree)
        
        return sdd
