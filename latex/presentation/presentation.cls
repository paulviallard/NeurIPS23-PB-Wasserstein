\ProvidesClass{presentation}[2023/10/25 Presentation]

\PassOptionsToClass{11pt}{beamer}
\PassOptionsToClass{fragile}{beamer}
\PassOptionsToClass{aspectratio=1610}{beamer}
\PassOptionsToClass{usepdftitle=false}{beamer}
\PassOptionsToClass{mathserif}{beamer}

\LoadClass{beamer}

\RequirePackage[LGR, OT1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage[default]{opensans}
\setbeamerfont{normal text}{family=Open Sans}

\RequirePackage{adjustbox}
\RequirePackage{booktabs}
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{amsthm}
\RequirePackage{amsfonts}
\RequirePackage{alphabeta}
\RequirePackage{datetime}
\RequirePackage[inline]{enumitem}
\RequirePackage{etoolbox}
\RequirePackage{fontawesome}
\RequirePackage{mathtools}
\RequirePackage[absolute,overlay]{textpos}
\RequirePackage{xspace}
\RequirePackage{pgfpages}
\RequirePackage{hyperref}
\RequirePackage{subcaption}
\captionsetup[subfigure]{labelformat=empty,justification=centering}
\RequirePackage{pgf}
\RequirePackage{caption}
\RequirePackage[misc]{ifsym}
\RequirePackage{tikz}
\RequirePackage{graphicx}
\RequirePackage{pifont}
\RequirePackage{thm-restate}
\RequirePackage{thmtools}
\RequirePackage{stmaryrd}
\RequirePackage{float}
\RequirePackage{multirow}
\RequirePackage{multicol}
\RequirePackage{longtable}
\RequirePackage{arydshln}
\RequirePackage{wasysym}
\RequirePackage{svg}
\RequirePackage{algorithm}
\RequirePackage{algpseudocode}
\RequirePackage{wrapfig}
\RequirePackage{lipsum} 


\RequirePackage[mode=buildnew]{standalone}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% graphicx options

\graphicspath{{figures/}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% font options

\newcommand{\cmark}{\ding{51}}
\newcommand{\xmark}{\ding{55}}

\newcommand{\blackbullet}{\textcolor{black}{\textbullet}~}
\newcommand{\redbullet}{\textcolor{red}{\textbullet}~}
\newcommand{\bluebullet}{\textcolor{blue}{\textbullet}~}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Tikz options

\usetikzlibrary{arrows.meta}
\tikzset{
    cross/.pic = {
    \draw[rotate = 45] (-#1,0) -- (#1,0);
    \draw[rotate = 45] (0,-#1) -- (0, #1);
    }
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Biblatex options

\RequirePackage[backend=biber, maxbibnames=10, maxcitenames=1, citestyle=authoryear-comp, bibstyle=authoryear, uniquelist=false, uniquename=false, dashed=false, natbib=true, sortcites=true, sorting=ynt]{biblatex}

\addbibresource{main.bib}
\renewcommand{\bibfont}{\normalfont\footnotesize}

% https://tex.stackexchange.com/questions/126226/how-do-i-instruct-fullcite-to-use-maxbibnames-rather-than-maxcitenames
\newcommand{\printpublication}[1]{\AtNextCite{\defcounter{maxnames}{99}}\fullcite{#1}}

\setlength{\bibhang}{0pt}
\setlength\bibitemsep{\baselineskip}

\renewbibmacro{in:}{}
\DeclareNameAlias{sortname}{given-family}

 % from \renewbibmacro*{date+extradate} in authoryear.bbx
 \renewbibmacro*{date+extradate}{}
  \renewbibmacro*{date}{%
    \iffieldundef{labelyear}
      {}
      {\printtext[parens]{%
         \iflabeldateisdate
           {\printfield{issue}%
            \setunit*{\addspace}%
            \printdateextra}
           {\printlabeldateextra}}}}

   % https://tex.stackexchange.com/questions/380299/change-position-of-year-field-in-authoryear-biblatex-style
  \renewbibmacro*{issue+date}{%
  \printtext[parens]{%
    \printfield{issue}%
    \setunit*{\addspace}%
     \printdate
     }%
  \newunit}

 % from \newbibmacro*{maintitle+booktitle} in standard.bbx
  \renewbibmacro*{maintitle+booktitle}{%
  \iffieldundef{booktitle}{}{\printfield{booktitle}}%
  \newunit}
  
  % from newbibmacro*{chapter+pages} in standard.bbx
  \renewbibmacro*{chapter+pages}{}
  % from newbibmacro*{note+pages} in standard.bbx  
  \renewbibmacro*{note+pages}{}

 % from \newbibmacro*{publisher+location+date} in standard.bbx
  \renewbibmacro*{publisher+location+date}{
  \printlist{publisher}
  \newunit
  \usebibmacro{date}
  \newunit}
  
  % from \newbibmacro*{journal+issuetitle} in standard.bbx
  \renewbibmacro*{journal+issuetitle}{%
  \usebibmacro{journal}%
  \newunit%
  \printfield{series}%
  \newunit
  \usebibmacro{volume+number+eid}%
  \newunit
  \usebibmacro{issue+date}%
  \newunit}

% https://github.com/plk/biblatex/issues/899
\DefineBibliographyStrings{english}{
  andothers = {\mkbibemph{et\addabbrvspace al\adddot}}
}

\DeclareFieldFormat*{title}{#1}
\DeclareFieldFormat*{booktitle}{\mkbibemph{#1}}
\DeclareListFormat*{publisher}{\mkbibemph{#1}}
\DeclareFieldFormat*{series}{\mkbibemph{#1}}

\DeclareCiteCommand{\citeauthor}{}
{
\ifdef{\theoldmaxnames}{}{\newcounter{oldmaxnames}}
\setcounter{oldmaxnames}{\themaxnames}
\setcounter{maxnames}{99}
\printnames{author}
\setcounter{maxnames}{\theoldmaxnames}
}{}{}

\DeclareCiteCommand{\citetitle}{}
{
\printfield{title}
}{}{}

\DeclareCiteCommand{\citedetails}{}
{\iffieldundef{booktitle}{
    \iffieldundef{journal}{
        \iffieldundef{note}{}{\printfield{note}}
    }{\printfield{journal}\iffieldundef{volume}{}{\printfield{volume}}}
}{\printfield{booktitle}}}{}{}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% xcolor options

\RequirePackage{xcolor}
% Paul Tol's  "Vibrant" color scheme
% https://personal.sron.nl/~pault/data/colourschemes.pdf

\definecolor{blue}{HTML}{0077BB}
\definecolor{cyan}{HTML}{33BBEE}
\definecolor{green}{HTML}{009988}
\definecolor{orange}{HTML}{EE7733}
\definecolor{red}{HTML}{CC3311}
\definecolor{magenta}{HTML}{EE3377}
\definecolor{grey}{HTML}{BBBBBB}
\definecolor{blockgrey}{RGB}{245, 245, 245}
\definecolor{blockblue}{RGB}{38, 62, 66}

\newcommand\white[1]{\textcolor{white}{#1}}
\newcommand\black[1]{\textcolor{black}{#1}}
\newcommand\blue[1]{\textcolor{blue}{#1}}
\newcommand\cyan[1]{\textcolor{cyan}{#1}}
\newcommand\green[1]{\textcolor{green}{#1}}
\newcommand\orange[1]{\textcolor{orange}{#1}}
\newcommand\red[1]{\textcolor{red}{#1}}
\newcommand\magenta[1]{\textcolor{magenta}{#1}}
\newcommand\grey[1]{\textcolor{grey}{#1}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tcolorbox options

\RequirePackage{tcolorbox}
\tcbuselibrary{skins}

\newtcolorbox{blueblock}[2][]{
    colback=blockgrey,
    colframe=blue,
    fonttitle=\bfseries,
    colbacktitle=blockgrey,
    enhanced, 
    coltitle=blue,
    attach boxed title to top left={yshift=-3mm,xshift=0.5cm}, 
    sharpish corners,
    title={#2}, #1, 
    boxed title style={frame code={}}
}
\newtcolorbox{redblock}[2][]{
    colback=blockgrey,
    colframe=red,
    fonttitle=\bfseries,
    colbacktitle=blockgrey,
    enhanced, 
    coltitle=red,
    attach boxed title to top left={yshift=-3mm,xshift=0.5cm}, 
    sharpish corners,
    title={#2}, #1, 
    boxed title style={frame code={}}
}
\newtcolorbox{blackblock}[2][]{
    colback=blockgrey,
    colframe=black,
    fonttitle=\bfseries,
    colbacktitle=blockgrey,
    enhanced, 
    coltitle=black,
    attach boxed title to top left={yshift=-4mm,xshift=0.5cm}, 
    sharpish corners,
    title={#2}, #1, 
    boxed title style={frame code={}}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% beamer options

\usetheme{CambridgeUS}
\usebeamercolor{structure}

\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{blocks}[default]

\setbeamercolor{subtitle}{fg=blockblue}
\setbeamercolor{title}{fg=blockblue}
\setbeamercolor{structure}{fg=red}
\setbeamercolor{section in head/foot}{bg=black,fg=white}
\setbeamercolor{frametitle}{bg=blockblue,fg=white} 
\setbeamercolor{block title}{bg=blockgrey, fg=black}
\setbeamercolor{block body}{bg=blockgrey}
\setbeamercolor{block canvas}{bg=black}

\setbeamerfont{frametitle}{size=\large}
\setbeamerfont{footnote}{size=\tiny}
\setbeamerfont{date}{size=\small}
\setbeamerfont{author}{size=\normalsize}
\setbeamerfont{institute}{size=\normalsize}

\setbeamercolor{bibliography entry author}{fg=red}
\setbeamercolor{bibliography entry title}{fg=black}
\setbeamercolor{bibliography entry note}{fg=red}

\setbeamertemplate{title page}{
  \vspace{2cm}
  \centering
    {\usebeamerfont{subtitle}\usebeamercolor[fg]{subtitle}\usebeamerfont{title}\textbf{\uppercase\expandafter{\inserttitle}}\par}%
    \ifx\insertsubtitle\@empty%
    \else%
      \vspace{0.05cm}%
      {\usebeamerfont{subtitle}\usebeamercolor[fg]{subtitle}\textbf{\uppercase\expandafter{\insertsubtitle}}\par}\fi
  \vspace{1.5cm}\par
  \usebeamerfont{author}\textbf{\insertauthor}\par
  \vspace{0.1cm}  
    \usebeamerfont{institute}\insertinstitute
  \vspace{1cm}
    \usebeamerfont{date}\insertdate
  \vskip0cm
}


\setbeamertemplate{headline}{
    \vskip0pt
}
\setbeamertemplate{frametitle}{%
    \nointerlineskip
    \begin{beamercolorbox}[sep=0.3cm,ht=1.0cm,wd=\paperwidth]{frametitle}
        \textbf{\uppercase\expandafter{\insertframetitle}}\vskip-0.0cm
    \end{beamercolorbox}
}

\setbeamertemplate{footline}{\hbox{\vspace{0.2cm}\hspace{0.95\textwidth}\insertframenumber/\inserttotalframenumber
}}

\setbeamertemplate{section in toc}{%
  \hspace{0.2cm}{\color{normal text.fg}\bf \inserttocsectionnumber.}
  \hspace{0.2cm}{\color{normal text.fg}\inserttocsection}
}

\renewcommand{\theenumi}{(\alph{enumi})} 
\newcommand{\labelenumi}{\textcolor{red}{\textbf{\theenumi}}}
\renewcommand{\theenumii}{{(}\roman{enumii}{)}} 
\newcommand{\labelenumii}{\textcolor{red}{\textbf{\theenumii}}}
\setbeamertemplate{enumerate item}{(\alph{enumi})}
\setbeamertemplate{enumerate subitem}{(\roman{enumii})}

\newenvironment{xframe}
{\begin{frame}[t,fragile,environment=xframe]}{\end{frame}}
\newenvironment{xplain}
{\begin{frame}[t,plain,fragile,environment=xplain]}{\end{frame}}
\newenvironment{xtitle}
{\setbeamercolor{bibliography entry author}{fg=white}\setbeamercolor{bibliography entry title}{fg=white}\setbeamercolor{bibliography entry note}{fg=white}\setbeamercolor{background canvas}{bg=blockblue,fg=white}\setbeamercolor{normal text}{fg=white}\usebeamercolor*{normal text}\begin{frame}[plain,fragile,environment=xtitle]\center\sc\Large}{\endcenter\setbeamercolor{bibliography entry author}{fg=red}\setbeamercolor{bibliography entry title}{fg=black}\setbeamercolor{bibliography entry note}{fg=red}\end{frame}}

\let\oitemize\itemize
\let\endoitemize\enditemize
\newenvironment{xitemize}
{\begin{oitemize}[label=$\textcolor{blockblue}{\bullet}$]}{\end{oitemize}}
\newenvironment{xitemize*}
{\begin{oitemize*}[label=$\textcolor{blockblue}{\bullet}$]}{\end{oitemize*}}
\let\itemize\xitemize
\let\enditemize\endxitemize

\newcommand{\xtableofcontents}[1]{
\setbeamercolor{background canvas}{bg=blockblue,fg=white}
\setbeamercolor{normal text}{bg=blockblue,fg=white}
\usebeamercolor*{normal text}
\begin{frame}[plain]{#1}
\tableofcontents
\end{frame}
\setbeamercolor{background canvas}{fg=black,bg=white}
\setbeamercolor{normal text}{fg=black}
\usebeamercolor*{normal text}
}

\let\oblock\block
\let\endoblock\endblock
\newenvironment{xblock}[1]{\ifthenelse{\equal{#1}{}}{\begin{oblock}{\vspace{-2mm}}}{\begin{oblock}{#1}}}{\end{oblock}}
\let\block\xblock
\let\endblock\endxblock
\let\xblock\relax
\let\endxblock\relax

\let\oscalebox\scalebox
\newcommand{\xscalebox}[2]{\oscalebox{#1}{\parbox{1.0\linewidth}{#2}}}
\let\scalebox\xscalebox

\renewcommand{\appendix}{%
\setbeamertemplate{footline}{}%
\renewenvironment{xframe}
{\begin{frame}[t,fragile,noframenumbering,environment=xframe]}{\end{frame}}
\renewenvironment{xtitle}
{\setbeamercolor{bibliography entry author}{fg=white}\setbeamercolor{bibliography entry title}{fg=white}\setbeamercolor{bibliography entry note}{fg=white}\setbeamercolor{background canvas}{bg=blockblue,fg=white}\setbeamercolor{normal text}{bg=blockblue,fg=white}\usebeamercolor*{normal text}\begin{frame}[plain,fragile,noframenumbering,environment=xtitle]\center\sc\Large}{\endcenter\setbeamercolor{bibliography entry author}{fg=red}\setbeamercolor{bibliography entry title}{fg=black}\setbeamercolor{bibliography entry note}{fg=red}\end{frame}}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% notes template

\setbeamercolor{note page}{bg=blockgrey, fg=black}
\setbeamertemplate{note page}{%
\insertvrule{1.0\paperheight}{note page.bg}%
\vskip-1.0\paperheight%
\nointerlineskip%
\vspace{0.8cm}\insertnote%
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% defcommand

\makeatletter
\newcommand{\defcommand}[2]{%
  \ifdefined#1%
    \renewcommand{#1}{#2}%
  \else%
    \newcommand{#1}{#2}%
  \fi
}
\makeatother