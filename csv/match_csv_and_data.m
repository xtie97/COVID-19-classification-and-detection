df = readtable('./SP_extra_info_opacity.csv');
filename_list = df.Filename;
label_list = df.label_positive;

Covid_file_list = dir(fullfile('../Dataset/SP/Covid/','*.png'));
Covid_filename_list = cell(length(Covid_file_list),1);
Covid_filename_cut_list = cell(length(Covid_file_list),1);
for ii = 1: length(Covid_file_list)
    Covid_filename_list{ii} = Covid_file_list(ii).name;
    Covid_filename_cut_list{ii} = Covid_file_list(ii).name(1:end-10);
end

NonCovid_file_list = dir(fullfile('../Dataset/SP/NonCovid/','*.png'));
NonCovid_filename_list = cell(length(NonCovid_file_list),1);
NonCovid_filename_cut_list = cell(length(NonCovid_file_list),1);
for ii = 1: length(NonCovid_file_list)
    NonCovid_filename_list{ii} = NonCovid_file_list(ii).name;
    NonCovid_filename_cut_list{ii} = NonCovid_file_list(ii).name(1:end-10);
end

%%
Loss_index = [];
for ii = 1: length(filename_list)
    filename = filename_list{ii};
    label = label_list(ii);
    filename = filename(strfind(filename, 'Covid')+6:end-10);
    if label > 0.5
        index = find(strcmp(Covid_filename_cut_list, filename));
        if ~isempty(index)
            filename_list{ii} = ['Covid/', Covid_filename_list{index}];
        else
            disp(ii);
            Loss_index = [Loss_index, ii];
        end
    elseif label < 0.5
        index = find(strcmp(NonCovid_filename_cut_list, filename));
        if ~isempty(index)
            filename_list{ii} = ['NonCovid/', NonCovid_filename_list{index}];
        else
            disp(ii);
            Loss_index = [Loss_index, ii];
        end
    end
end

df.Filename = filename_list; 
df(Loss_index,:) = [];
writetable(df, 'SP_new_extra_info_opacity.csv', 'WriteRowNames',false); 