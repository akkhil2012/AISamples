from neo4j import GraphDatabase, Result


class Neo4jPopulator:
    def __init__(self, uri, username, password):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    async def populate_graph(self, query: str) -> Result:
        with self.driver.session() as session:
            result = session.run(query)
            return result

    async def close(self):
        self.driver.close()
