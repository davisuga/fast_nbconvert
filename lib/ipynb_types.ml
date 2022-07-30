(* type cell_type = Code | Markdown

   let cell_type_to_yojson = function Code -> "code" | Markdown -> "markdown" *)

type cell = { cell_type : string; source : string list }
[@@deriving yojson { strict = false }]

type python_notebook = {
  cells : cell list;
      (* metadata : python_notebook_metadata; *)
      (* nbformat : int; *)
      (* nbformatMinor : int; *)
}
[@@deriving yojson { strict = false }]

(* and cell_metadata = { id : string } [@@deriving yojson { strict = false }] *)

(* and python_notebook_metadata = { colab : colab; kernelspec : kernelspec }
   [@@deriving yojson { strict = false }] *)

(* and provenance = { fileID : string; timestamp : int }
   [@@deriving yojson { strict = false }] *)

(* and kernelspec = { displayName : string; name : string }
   [@@deriving yojson { strict = false }] *)
