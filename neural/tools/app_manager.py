from dash import Dash, html, dcc
from dash.dependencies import Input, Output
from neural.tools.graph_factory import make_graph
from threading import Thread


def create_div_for(namespace, graph_type):
    name = namespace + "-" + graph_type
    return html.Div(
        children=[
            html.H1(children=name),
            dcc.Graph(
                id=name,
            ),
            dcc.Interval(id="interval_" + name, interval=20 * 1000, n_intervals=0),
        ]
    )


class AppManager:
    def __init__(self, app_name):
        self._app = Dash(app_name)

        self._graphs = {}
        self._handlers = {}
        self._active = False

    def add(self, namespace, graph_type):
        pieces = namespace.split("-")
        current = self._handlers
        for p in pieces:
            if p not in current:
                current[p] = {}
            current = current[p]
        final_name = pieces[-1]

        top_level = pieces[0]
        if top_level not in self._graphs:
            self._graphs[top_level] = {}
        if graph_type not in self._graphs[top_level]:
            self._graphs[top_level][graph_type] = make_graph(graph_type)

        current[graph_type] = lambda data: self._graphs[top_level][graph_type].push(
            final_name, data
        )

    def forward_to_handler(self, namespace, graph_type, data):
        pieces = namespace.split("-")
        current = self._handlers
        for p in pieces:
            current = current[p]
        current[graph_type](data)

    def render(self):
        def _helper():
            divs = []
            for namespace, graphs in self._graphs.items():
                for g_type, graph in graphs.items():
                    divs.append(create_div_for(namespace, g_type))
            return divs

        app = self._app
        app.layout = html.Div(children=_helper())
        print(f"app.layout={app.layout}")

        def _add_callback(name, graph):
            @app.callback(
                Output(name, "figure"), Input("interval_" + name, "n_intervals")
            )
            def _callback(n):
                return graph.render()

        for namespace, graphs in self._graphs.items():
            for graph_type, graph in graphs.items():
                _add_callback(namespace + "-" + graph_type, graph)

        print(f"running server")

        def _run():
            app.run_server()

        t = Thread(target=_run, daemon=True)
        t.start()

        print(f"server is running")
        self._active = True
