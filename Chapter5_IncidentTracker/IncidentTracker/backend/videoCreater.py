from manim import *

# install manim

class IncidentTrackerFlow(Scene):
    def construct(self):
        # Create nodes
        ui = RoundedRectangle(
            corner_radius=0.5,
            height=2, width=4,
            fill_color=BLUE_D, fill_opacity=0.8,
            stroke_color=BLUE_A
        ).to_edge(LEFT)
        ui_text = Text("Streamlit UI", color=WHITE).move_to(ui.get_center())
        
        llm = Circle(
            radius=1.5,
            fill_color=GREEN_D, fill_opacity=0.8,
            stroke_color=GREEN_A
        ).shift(RIGHT * 2)
        llm_text = Text("LLM\nQuery\nProcessor", color=WHITE, font_size=24).move_to(llm.get_center())
        
        servers = VGroup()
        server_icons = []
        server_texts = ["Confluence", "Teams", "Outbox", "Local Disk"]
        server_colors = [YELLOW_D, PURPLE_D, ORANGE, PINK]
        
        for i in range(4):
            server = RoundedRectangle(
                corner_radius=0.3,
                height=1.5, width=3,
                fill_color=server_colors[i], fill_opacity=0.8,
                stroke_color=server_colors[i] + "_A"
            ).shift(RIGHT * 5 + UP * (1.5 - i))
            text = Text(server_texts[i], font_size=18).move_to(server.get_center())
            servers.add(server, text)
            server_icons.append(server)
        
        results = RoundedRectangle(
            corner_radius=0.5,
            height=2, width=4,
            fill_color=RED_D, fill_opacity=0.8,
            stroke_color=RED_A
        ).to_edge(RIGHT)
        results_text = Text("Results &\nPDF Export", color=WHITE, font_size=20).move_to(results.get_center())
        
        # Add all to scene
        self.play(
            Create(ui),
            Write(ui_text),
            run_time=1
        )
        self.play(
            Create(llm),
            Write(llm_text),
            run_time=1
        )
        self.play(
            *[Create(server) for server in servers],
            run_time=2
        )
        self.play(
            Create(results),
            Write(results_text),
            run_time=1
        )
        
        # Create arrows and animate flow
        arrow1 = Arrow(ui.get_right(), llm.get_left(), buff=0.2)
        self.play(GrowArrow(arrow1))
        
        arrows = []
        for i, server in enumerate(server_icons):
            arrow = Arrow(
                llm.get_right(),
                server.get_left(),
                buff=0.2,
                color=server_colors[i]
            )
            self.play(
                GrowArrow(arrow),
                server.animate.set_fill(opacity=1),
                run_time=0.5
            )
            arrows.append(arrow)
            
            # Animate data returning to LLM
            return_arrow = Arrow(
                server.get_left(),
                llm.get_right(),
                buff=0.2,
                color=server_colors[i],
                stroke_width=3
            )
            self.play(
                GrowArrow(return_arrow),
                run_time=0.5
            )
            self.remove(return_arrow)
        
        # Final processing and results
        final_arrow = Arrow(llm.get_bottom(), results.get_left(), buff=0.2)
        self.play(
            GrowArrow(final_arrow),
            results.animate.set_fill(opacity=1)
        )
        
        # Add title and description
        title = Text("Incident Tracker Workflow", font_size=36).to_edge(UP)
        self.play(Write(title))
        
        self.wait(2)

# To run this script, you'll need to install manim first:
# pip install manim
# Then run: manim -pql videoCreater.py IncidentTrackerFlow