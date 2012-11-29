/*!
 * @file GraphCutMultiLabel.cpp
 * @author a_hasimoto
 * @date Date Created: 2012/Nov/14
 * @date Last Change: 2012/Nov/27.
 */
#include "GraphCutMultiLabel.h"

using namespace skl;
typedef Graph<int,int,int> Graph_i;

int GraphCutMultiLabel::ENOUGH_SMALL_VALUE = 1;
const int GraphCutMultiLabel::MAX_TERM_VALUE = 65536;
int GraphCutMultiLabel::SET_SYMMETRY_VALUE = -INT_MAX;

GraphCutMultiLabel::Node::Node(size_t label_num, int default_data_term)
	:layers(label_num-1),data_terms(label_num,default_data_term)
{
}

size_t GraphCutMultiLabel::Node::getLabel(const cv::Ptr<Graph_i> graph)const{
	for(size_t l = 0; l < layers.size(); l++){
		if( graph->what_segment( layers[l], Graph_i::SINK ) ) return l;
	}
	return layers.size();
}

GraphCutMultiLabel::Edge::Edge(size_t label_num,const cv::Ptr<Node> __from, const cv::Ptr<Node> __to)
	:smoothing_terms(label_num,ENOUGH_SMALL_VALUE),
	 smoothing_terms_inv(label_num,ENOUGH_SMALL_VALUE),
	 from(__from),to(__to)
{
}

/*!
 * @brief デフォルトコンストラクタ
 */
GraphCutMultiLabel::GraphCutMultiLabel(size_t __label_num):_label_num(__label_num){
	assert(_label_num>1);
}


/*!
 * @brief デストラクタ
 */
GraphCutMultiLabel::~GraphCutMultiLabel(){

}

cv::Ptr<Graph_i> GraphCutMultiLabel::constructGraph(){
	assert(_label_num>1);
	cv::Ptr<Graph_i> pGraph = new Graph_i(_node_num(),_edge_num());

	// vertical edges
	for(size_t n=0;n<_nodes.size();n++){
		cv::Ptr<Node> pNode = _nodes[n];
		assert(pNode->label_num()==_label_num);
		pNode->layers[0] = pGraph->add_node();

		if(_label_num==2){
			// there are no layer node when _label_num==2.
			pGraph->add_tweights(
					pNode->layers[0],
					pNode->data_terms[0],
					pNode->data_terms[1]);
			continue;
		}

		pGraph->add_tweights(
				pNode->layers[0],
				pNode->data_terms[0],
				MAX_TERM_VALUE);
		size_t l;
		for(l = 1; l < pNode->layers.size(); l++){
			pNode->layers[l] = pGraph->add_node();
			pGraph->add_edge(
					pNode->layers[l-1],
					pNode->layers[l],
					pNode->data_terms[l],
					MAX_TERM_VALUE);
		}
		pGraph->add_tweights(
				pNode->layers[l-1],
				MAX_TERM_VALUE,
				pNode->data_terms[l]);
	}

	// horizontal edges
	for(size_t e = 0; e < _edges.size(); e++){
		cv::Ptr<Edge> pEdge = _edges[e];
		cv::Ptr<Node> from = pEdge->from;
		cv::Ptr<Node> to = pEdge->to;
		for(size_t l=0;l<from->layers.size();l++){
			pGraph->add_edge(
				from->layers[l],
				to->layers[l],
				pEdge->smoothing_terms[l],
				pEdge->smoothing_terms_inv[l]);
		}
	}

	return pGraph;
}

void GraphCutMultiLabel::addEdge(NodeID node1, NodeID node2, int smoothing_term, int smoothing_term_inv){
	assert(smoothing_term >= 0);
	assert(node1 < _nodes.size());
	assert(node2 < _nodes.size());
	assert(node1 != node2);
	if(smoothing_term_inv == SET_SYMMETRY_VALUE){
		smoothing_term_inv = smoothing_term;
	}
	cv::Ptr<Edge> edge = new Edge(_label_num,_nodes[node1],_nodes[node2]);
	edge->smoothing_terms.assign(_label_num-1,smoothing_term);
	edge->smoothing_terms_inv.assign(_label_num-1,smoothing_term_inv);
}

int GraphCutMultiLabel::compute(std::vector<size_t>& labels){
	labels.resize(_nodes.size());
	_graph = constructGraph();
	int total_cost = _graph->maxflow();
	for(size_t n =0; n < _nodes.size(); n++){
		labels[n] = _nodes[n]->getLabel(_graph);
	}
	return total_cost;
}
