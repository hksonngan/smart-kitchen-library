/*!
 * @file GraphCutMultiLabel.h
 * @author a_hasimoto
 * @date Date Created: 2012/Nov/14
 * @date Last Change:2012/Nov/30.
 */
#ifndef __SKL_GRAPH_CUT_MULTI_LABEL_H__
#define __SKL_GRAPH_CUT_MULTI_LABEL_H__

#include "skl.h"
#include <cv.h>

namespace skl{

/*!
 * @class GraphCutMultiLabel
 * @brief Grobal Minimization of Energy by GraphCut algorithm with Multi Label. The labels must be ordered.
 */
class GraphCutMultiLabel{
	private:
		static int ENOUGH_SMALL_VALUE;
		static int SET_SYMMETRY_VALUE;
	protected:
		typedef Graph<int,int,int> Graph_i;
	public:
		static const int MAX_TERM_VALUE;
		typedef size_t NodeID;
		enum Terminal{
			SOURCE=0,
			SINK=1
		};

		/*
		 * Nodes corresponds to an element with multiple labels
		 * */
		class Node{
			public:
				Node(size_t label_num, int default_data_term=ENOUGH_SMALL_VALUE);
				Node(const Node& other);
				~Node(){}
				std::vector<Graph_i::node_id> layers;
				std::vector<int> data_terms;
				size_t getLabel(const cv::Ptr<Graph_i> graph)const;
				inline size_t label_num()const{return data_terms.size();}
		};

		/*
		 * Edges between Node class instances.
		 * */
		class Edge{
			public:
				Edge(size_t label_num, const cv::Ptr<Node> from, const cv::Ptr<Node> to);
				~Edge(){}
				inline void setSmoothingTerm(int value){
					assert(value>=0);
					smoothing_terms.assign(smoothing_terms.size(),value);
				}
				std::vector<int> smoothing_terms;
				std::vector<int> smoothing_terms_inv;
				const cv::Ptr<Node> from;
				const cv::Ptr<Node> to;
		};

	public:
		GraphCutMultiLabel(size_t label_num);


		virtual ~GraphCutMultiLabel();
		int compute(std::vector<size_t>& labels);

		inline NodeID addNode(int term_between_labels=ENOUGH_SMALL_VALUE){
			_nodes.push_back(cv::Ptr<Node>(new Node(_label_num,term_between_labels)));
			return _nodes.size()-1;
		}
		inline void setDataTerm(NodeID n, int data_term_src, int data_term_sink){
			assert(n < _nodes.size());
			assert(!_nodes[n].empty());
			assert(_nodes[n]->label_num()==_label_num);
			assert(_nodes[n]->data_terms.size()>0);
			_nodes[n]->data_terms[0] = data_term_src;
			_nodes[n]->data_terms[_nodes[n]->data_terms.size()-1] = data_term_sink;
		}
		NodeID addNode(int data_term_src,int data_term_sink,int term_between_labels=ENOUGH_SMALL_VALUE);

		void addEdge(NodeID node1, NodeID node2, int smoothing_term, int smoothing_term_inv = SET_SYMMETRY_VALUE);

		inline size_t label_num()const{return _label_num;}
		inline const std::vector<cv::Ptr<Node> >& nodes()const{return _nodes;}
		inline const std::vector<cv::Ptr<Edge> >& edges()const{return _edges;}

	protected:
		size_t _label_num;
		std::vector<cv::Ptr<Node> > _nodes;
		std::vector<cv::Ptr<Edge> > _edges;
		cv::Ptr<Graph_i> _graph;
		cv::Ptr<Graph_i> constructGraph();
	private:
		// node num for _graph.
		inline size_t _node_num()const{
			return (_label_num-1) * _nodes.size();
		}
		// edge num for _edge
		inline size_t _edge_num()const{
			return (_label_num-2)*_nodes.size() + (_label_num-1)*_edges.size();
		}
};

} // skl

#endif // __SKL_GRAPH_CUT_MULTI_LABEL_H__

