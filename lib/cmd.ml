open Cmdliner

let files_arg =
  Arg.(
    non_empty
    & pos_all file []
    & info [] ~docv:"FILES" ~doc:"The paths to the source files")

let make_file_flag =
  Arg.(
    last
    & vflag_all [ false ]
        [
          ( true,
            info ~doc:"Creates a file instead of printing to stdout"
              [ "make-file"; "mk" ] );
        ])

let command_term =
  Term.(
    const (fun paths create_file ->
        List.map (Parse.make_py_file ~create_file) paths |> ignore)
    $ files_arg
    $ make_file_flag)

let main_command =
  let command_documentation =
    Cmd.info "Fast NB Converter"
      ~doc:"Parses Jupyter Notebooks into code and vice versa!"
  in

  Cmd.v command_documentation command_term

let run () = exit (Cmd.eval main_command)
